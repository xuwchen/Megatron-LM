# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Kimi's fused latent up-projection must retain full RMSNorm and linear semantics."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TERMSNormDuplicatedLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.tensor_parallel.gtp_api import (
    GTP_CONFIG,
    is_gtp_param,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _torchrun_dist_init,
    reset_gtp_globals,
)


@pytest.mark.parametrize("rmsnorm_impl", ["te", "torch"])
@pytest.mark.parametrize("per_token_loss", [False, True])
@pytest.mark.parametrize("output_size", [80, 128])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_latent_projection_matches_full_matrix(
    output_size, bias, per_token_loss, rmsnorm_impl, monkeypatch
):
    """Check padding, replicated norm/bias, forward, dgrad, and global wgrad."""
    # This standalone layer has no training driver to configure global GTP state.
    monkeypatch.setattr(GTP_CONFIG, "calculate_per_token_loss", per_token_loss)
    world = dist.get_world_size()
    if world != 4:
        pytest.skip("Run with four torchrun ranks")
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(gtp_remat_size=world)
    _set_random_seed(1234)
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        normalization="RMSNorm",
        rmsnorm_impl=rmsnorm_impl,
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        add_bias_linear=bias,
        gtp_weight_remat_size=world,
    )
    layer = TERMSNormDuplicatedLinear(
        64,
        output_size,
        parallel_mode="duplicated",
        config=config,
        init_method=config.init_method,
        bias=bias,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_group=pgc.tp,
        gtp_remat_group=pgc.gtp_remat,
        gtp_replica_group=pgc.dp_cp,
    )
    assert is_gtp_param(layer.weight)
    assert not is_gtp_param(layer.layer_norm_weight)
    assert layer.layer_norm_weight.shape == (64,)
    assert layer.out_features == output_size
    if bias:
        assert not is_gtp_param(layer.bias)
        assert layer.bias.shape == (output_size,)

    shards = [torch.empty_like(layer.weight) for _ in range(world)]
    dist.all_gather(shards, layer.weight.detach(), group=pgc.gtp_remat)
    weight = torch.cat(shards)[:output_size].detach().requires_grad_()
    norm = layer.layer_norm_weight.detach().clone().requires_grad_()
    ref_bias = layer.bias.detach().clone().requires_grad_() if bias else None
    torch.manual_seed(100 + dist.get_rank())
    x = torch.randn(16, 1, 64, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    layer.weight.main_grad = torch.zeros_like(layer.weight)

    actual, _ = layer(x)
    expected = F.linear(F.rms_norm(ref_x, (64,), norm, config.layernorm_epsilon), weight, ref_bias)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    grad = torch.randn_like(expected)
    actual.backward(grad)
    expected.backward(grad)
    wait_for_gtp_grad_reduction_on_current_stream()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(layer.layer_norm_weight.grad, norm.grad, atol=2e-5, rtol=2e-5)
    if bias:
        torch.testing.assert_close(layer.bias.grad, ref_bias.grad, atol=2e-5, rtol=2e-5)
    dist.all_reduce(weight.grad, group=pgc.gtp_remat)
    # Default GTP returns the DP mean; token-loss mode returns a sum for the
    # training driver to normalize by the global token count.
    if not per_token_loss:
        weight.grad.div_(world)
    padded = F.pad(weight.grad, (0, 0, 0, layer.weight.pad_length))
    expected_shard = padded.chunk(world)[pgc.gtp_remat.rank()]
    torch.testing.assert_close(layer.weight.main_grad, expected_shard, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("zero_centered", [False, True])
def test_native_bf16_slim_projection_gtp(zero_centered):
    """Published slim latent512 -> H1024; independent HF order and GTP mean wgrad."""
    world = dist.get_world_size()
    # Construction, gather, mean reduction and reference shard selection below
    # all use the actual world size. Exercise both the GTP4 recipe and 8-GPU CI.
    assert world in (4, 8), "Run this regression with four or eight torchrun ranks"
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(gtp_remat_size=world)
    _set_random_seed(20260821)
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        normalization="RMSNorm",
        rmsnorm_impl="torch",
        layernorm_zero_centered_gamma=zero_centered,
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
        gtp_weight_remat_size=world,
    )
    layer = TERMSNormDuplicatedLinear(
        512,
        1024,
        parallel_mode="duplicated",
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_group=pgc.tp,
        gtp_remat_group=pgc.gtp_remat,
        gtp_replica_group=pgc.dp_cp,
    ).cuda()
    assert is_gtp_param(layer.weight) and not is_gtp_param(layer.layer_norm_weight)
    with torch.no_grad():
        layer.layer_norm_weight.add_(torch.randn_like(layer.layer_norm_weight) * 0.05)
    shards = [torch.empty_like(layer.weight) for _ in range(world)]
    dist.all_gather(shards, layer.weight.detach(), group=pgc.gtp_remat)
    weight = torch.cat(shards)[:1024].detach().requires_grad_()
    norm = layer.layer_norm_weight.detach().clone().requires_grad_()
    torch.manual_seed(20260821 + dist.get_rank())
    x = torch.randn(128, 1, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
    xf = rx.float()
    normalized = (
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + config.layernorm_epsilon)
    ).to(rx.dtype)
    expected = F.linear(normalized * (norm + 1 if zero_centered else norm), weight)
    actual, _ = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    wait_for_gtp_grad_reduction_on_current_stream()
    torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
    torch.testing.assert_close(layer.layer_norm_weight.grad, norm.grad, rtol=0, atol=0)
    global_grad = weight.grad.float()
    dist.all_reduce(global_grad, group=pgc.gtp_remat)
    global_grad.div_(world)
    expected_shard = global_grad.chunk(world)[pgc.gtp_remat.rank()]
    a, b = layer.weight.main_grad.double().flatten(), expected_shard.double().flatten()
    assert F.cosine_similarity(a[None], b[None]).item() >= 1 - 1e-3
    assert (2 * (a * b).sum() / (a.square().sum() + b.square().sum())).item() >= 1 - 1e-3
    assert ((a - b).norm() / b.norm()).item() <= 0.005
