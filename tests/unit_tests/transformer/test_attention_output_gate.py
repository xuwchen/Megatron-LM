# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Saved-tensor and parity regressions for the elementwise MLA output gate helper.

The dev branch carries a larger file of this name (output-gate math and exact
upstream rounding regressions for Attention, MultiLatentAttention, absorbed MLA
and KDA). Those depend on dev-only code paths; this file ports only the cases
that cover ``apply_elementwise_attention_output_gate_before_cast``: the eager
composition (``fusion=False``) and the Triton fusion (``fusion=True``) produce the
same values and gradients, and the fusion keeps only the two inputs for backward.
Numerical parity cases require real CUDA; shape-validation cases run on CPU.
"""

import pytest
import torch

from megatron.core.fusions.fused_attention_output_gate import HAVE_TRITON
from megatron.core.transformer.attention_output_gate import (
    apply_attention_output_gate,
    apply_elementwise_attention_output_gate_before_cast,
)
from tests.unit_tests.test_utilities import Utils

HY4_NUM_HEADS = 64
HY4_VALUE_HEAD_DIM = 256
DTYPES = [torch.float32, torch.bfloat16]


@pytest.fixture
def cuda_device():
    assert torch.cuda.is_available(), "CUDA regression tests require a real GPU."
    torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
    return torch.device("cuda")


def _inputs(dtype, device):
    shape = (2, 1, HY4_NUM_HEADS * HY4_VALUE_HEAD_DIM)
    values = torch.linspace(-2.0, 2.0, 2 * shape[-1], device=device).to(dtype).view(shape)
    logits = torch.linspace(-8.0, 8.0, 2 * shape[-1], device=device).to(dtype)
    return values, logits.view(shape)


def _assert_output_and_vjp(actual_fn, reference_fn, values, logits):
    actual_values = values.detach().clone().requires_grad_(True)
    actual_logits = logits.detach().clone().requires_grad_(True)
    reference_values = values.detach().clone().requires_grad_(True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    actual = actual_fn(actual_values, actual_logits)
    expected = reference_fn(reference_values, reference_logits)
    gradient = torch.linspace(-1.0, 1.0, actual.numel(), device=actual.device)
    gradient = gradient.to(actual.dtype).view_as(actual)
    actual_grads = torch.autograd.grad(actual, (actual_values, actual_logits), gradient)
    expected_grads = torch.autograd.grad(expected, (reference_values, reference_logits), gradient)
    assert actual.dtype == values.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    return actual


def _eager_gate(x, g):
    return apply_attention_output_gate(x, g, "elementwise", cast_mode="before")


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("gate_dtype", DTYPES, ids=["gate-fp32", "gate-bf16"])
def test_elementwise_gate_eager_path_is_the_eager_composition(dtype, gate_dtype, cuda_device):
    values, logits = _inputs(dtype, cuda_device)
    logits = logits.to(gate_dtype)
    output = _assert_output_and_vjp(
        lambda x, g: apply_elementwise_attention_output_gate_before_cast(x, g, fusion=False),
        _eager_gate,
        values,
        logits,
    )
    assert type(output.grad_fn).__name__ == "MulBackward0"


@pytest.mark.skipif(not HAVE_TRITON, reason="attention_output_gate_fusion needs Triton")
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("gate_dtype", DTYPES, ids=["gate-fp32", "gate-bf16"])
def test_elementwise_gate_fusion_matches_eager_and_saves_only_inputs(
    dtype, gate_dtype, cuda_device
):
    # The MLA elementwise gate keeps the eager rounding and VJP but must not keep the
    # FP32 sigmoid output and the cast scale alive until backward: 384 MiB per MLA
    # layer and microbatch at 8K tokens x 12288 channels, for every live microbatch.
    values, logits = _inputs(dtype, cuda_device)
    logits = logits.to(gate_dtype)
    _assert_output_and_vjp(
        lambda x, g: apply_elementwise_attention_output_gate_before_cast(x, g, fusion=True),
        _eager_gate,
        values,
        logits,
    )
    gated_values = values.detach().clone().requires_grad_(True)
    gated_logits = logits.detach().clone().requires_grad_(True)
    output = apply_elementwise_attention_output_gate_before_cast(
        gated_values, gated_logits, fusion=True
    )
    assert type(output.grad_fn).__name__ == "_FusedElementwiseGateBeforeCastBackward"
    saved = output.grad_fn.saved_tensors
    assert len(saved) == 2
    assert saved[0].data_ptr() == gated_values.data_ptr()
    assert saved[1].data_ptr() == gated_logits.data_ptr()
    # Without grad the helper is the eager composition itself.
    with torch.no_grad():
        eager = _eager_gate(values, logits)
        fused = apply_elementwise_attention_output_gate_before_cast(values, logits, fusion=True)
        assert torch.equal(fused, eager)


def test_elementwise_gate_before_cast_rejects_invalid_shapes():
    values = torch.zeros(2, 1, 64)
    with pytest.raises(ValueError):
        apply_elementwise_attention_output_gate_before_cast(values, torch.zeros(2, 1, 16))
