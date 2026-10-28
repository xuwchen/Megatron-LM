# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Capture complete PP1 depth-source lifetimes across microbatches and updates."""

import copy

import pytest
import torch
from torch import nn

from megatron.core.transformer.attention_residual import (
    AttentionResidual,
    AttnResStageSources,
    is_attn_res_block_start,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class DepthStack(nn.Module):
    def __init__(self, dtype, impl):
        super().__init__()
        self.config = TransformerConfig(
            num_layers=7,
            hidden_size=1024,
            num_attention_heads=16,
            enable_attention_residuals=True,
            attn_res_impl=impl,
            attn_res_block_layers=2,
            is_hybrid_model=True,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            cuda_graph_impl="full_iteration",
            cuda_graph_modules=[],
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        self.aggregates = nn.ModuleList([AttentionResidual(self.config) for _ in range(8)])
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.randn(1024, dtype=dtype) * 0.1) for _ in range(7)]
        )
        with torch.no_grad():
            for module in self.aggregates:
                module.pseudo_query.normal_(std=0.01)

    def forward(self, x):
        state, partial = AttnResStageSources.enter(
            self.config,
            x,
            layers_before=0,
            pp_rank=0,
            vp_stage=None,
            microbatch_id=None,
            pre_process=True,
        )
        for index, scale in enumerate(self.scales):
            start = is_attn_res_block_start(index + 1, self.config.attn_res_block_layers)
            if start:
                state.append_block_start(partial)
            values = list(state.graph_sources) if start else [*state.graph_sources, partial]
            branch = torch.tanh(self.aggregates[index](values) * scale)
            partial = branch if start else partial + branch
        return self.aggregates[-1](state.exit_aggregate_values(partial))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("impl", ["eager", "fla"])
def test_attn_res_full_schedule_capture_replay(dtype, impl):
    if impl == "fla":
        pytest.importorskip("fla.ops.attnres")
    torch.manual_seed(123)
    model = DepthStack(dtype, impl).cuda()
    reference = copy.deepcopy(model)
    inputs = [
        torch.randn(128, 1, 1024, device='cuda', dtype=dtype, requires_grad=True) for _ in range(2)
    ]
    ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    upstream = [torch.randn_like(x) for x in inputs]
    for stack in (model, reference):
        for parameter in stack.parameters():
            parameter.grad = torch.zeros_like(parameter)
    for x in inputs + ref_inputs:
        x.grad = torch.zeros_like(x)

    def step(stack, xs):
        for parameter in stack.parameters():
            parameter.grad.zero_()
        outputs = []
        for x, grad_output in zip(xs, upstream):
            x.grad.zero_()
            output = stack(x)
            output.backward(grad_output)
            outputs.append(output)
        return outputs

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step(model, inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_outputs = step(model, inputs)
    previous = None
    for iteration in range(5):
        with torch.no_grad():
            for x, rx, upstream_tensor in zip(inputs, ref_inputs, upstream):
                x.normal_()
                rx.copy_(x)
                upstream_tensor.normal_()
        graph.replay()
        expected = step(reference, ref_inputs)
        pairs = list(zip(graph_outputs, expected))
        pairs += [(x.grad, rx.grad) for x, rx in zip(inputs, ref_inputs)]
        pairs += [(p.grad, q.grad) for p, q in zip(model.parameters(), reference.parameters())]
        for actual, wanted in pairs:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-5)
        if previous is not None:
            assert not torch.equal(graph_outputs[0], previous), 'Replay returned stale output'
        previous = graph_outputs[0].clone()
        with torch.no_grad():
            for p, q in zip(model.parameters(), reference.parameters()):
                p.add_(p.grad, alpha=-1e-5)
                q.add_(q.grad, alpha=-1e-5)
                torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-5)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "overrides",
    [
        {"cuda_graph_impl": "local"},
        {"pipeline_model_parallel_size": 2},
        {"attn_res_impl": "compile"},
        {"attn_res_impl": "fla", "cuda_graph_impl": "local"},
        {"attn_res_impl": "fla", "pipeline_model_parallel_size": 2},
    ],
)
def test_unverified_attn_res_graph_modes_remain_rejected(overrides):
    kwargs = dict(
        num_layers=4,
        hidden_size=16,
        num_attention_heads=4,
        pipeline_dtype=torch.float32,
        enable_attention_residuals=True,
        attn_res_block_layers=2,
        cuda_graph_impl="full_iteration",
        cuda_graph_modules=[],
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match="Attention Residuals graphs require"):
        TransformerConfig(**kwargs)
