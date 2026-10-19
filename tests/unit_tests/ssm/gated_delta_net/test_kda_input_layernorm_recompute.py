# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KDA input-layernorm recompute wiring with real projection modules.

This file also runs with ``--confcutdir=tests/unit_tests/ssm/gated_delta_net``;
it owns its distributed setup and does not change NCCL environment settings.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    owns_default_group = not dist.is_initialized()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        yield
    finally:
        Utils.destroy_model_parallel()
        if owns_default_group and dist.is_initialized():
            dist.destroy_process_group()
            Utils.store = None


@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA KDA is required")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "f_rank,gate_rank",
    [(None, None), (32, None), (None, 32), (32, 32)],
    ids=["legacy-fused", "low-rank-f", "low-rank-gate", "low-rank-both"],
)
def test_kda_input_layernorm_recompute_preserves_direct_projection_inputs(
    model_parallel, f_rank, gate_rank
):
    config = TransformerConfig(
        hidden_size=128,
        num_layers=1,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        kda_f_lora_rank=f_rank,
        kda_gate_lora_rank=gate_rank,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        normalization="RMSNorm",
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        experimental_attention_variant="kda",
        is_hybrid_model=True,
        transformer_impl="transformer_engine",
    )
    spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    module = KimiDeltaAttention(
        config,
        submodules=spec.submodules,
        layer_number=1,
        conv_init=1.0,
        pg_collection=ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        ),
    ).cuda()

    module.set_for_recompute_input_layernorm()

    consumers = [module.in_proj, module.beta_proj]
    if f_rank is not None or gate_rank is not None:
        consumers.append(module.f_proj if f_rank is None else module.f_a_proj)
        consumers.append(module.g_proj if gate_rank is None else module.g_a_proj)
    assert all(projection.save_original_input for projection in consumers)
    for name in ("f_b_proj", "g_b_proj"):
        if hasattr(module, name):
            assert not getattr(module, name).save_original_input
