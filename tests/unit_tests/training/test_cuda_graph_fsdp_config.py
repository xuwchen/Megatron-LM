# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Megatron-FSDP CUDA graph argument validation."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _validate_cuda_graph_config
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    _get_allocator_namespace,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.training.arguments import (
    _get_cuda_graph_recompute_overlap,
    _validate_megatron_fsdp_cuda_graph_buffers,
)
from megatron.training.training import wrap_model_chunks_with_ddp

_RECOMPUTE_MODULES = ["shared_experts", "moe_act", "mlp", "layernorm", "moe"]


@pytest.mark.parametrize(
    ("cuda_graph_modules", "expected_overlap"),
    [
        pytest.param([], _RECOMPUTE_MODULES, id="whole-layer"),
        pytest.param([CudaGraphModule.mlp], ["mlp", "layernorm"], id="mlp"),
        pytest.param(
            [CudaGraphModule.moe],
            ["shared_experts", "moe_act", "layernorm", "moe"],
            id="moe",
        ),
        pytest.param([CudaGraphModule.moe_router], ["shared_experts", "moe"], id="moe-router"),
        pytest.param(
            [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
            ["shared_experts", "moe"],
            id="moe-router-and-preprocess",
        ),
        pytest.param([CudaGraphModule.attn], [], id="attention"),
        pytest.param(
            [CudaGraphModule.mlp, CudaGraphModule.moe],
            _RECOMPUTE_MODULES,
            id="mlp-and-moe",
        ),
    ],
)
def test_recompute_overlap_uses_normalized_cuda_graph_enums(
    cuda_graph_modules, expected_overlap
):
    """Each normalized capture scope reports only recompute modules it captures."""
    assert (
        _get_cuda_graph_recompute_overlap(cuda_graph_modules, _RECOMPUTE_MODULES)
        == expected_overlap
    )


@pytest.mark.parametrize(
    ("cuda_graph_modules", "expected_overlap"),
    [
        pytest.param([], ["gdn"], id="whole-layer"),
        pytest.param([CudaGraphModule.attn], ["gdn"], id="attention"),
        pytest.param([CudaGraphModule.moe_router], [], id="router-only"),
    ],
)
def test_gdn_recompute_overlap_requires_attention_graph_opt_in(
    cuda_graph_modules, expected_overlap
):
    """GDN recompute overlaps only scopes that capture opted-in GDN attention."""
    assert (
        _get_cuda_graph_recompute_overlap(
            cuda_graph_modules, ["gdn"], captures_gdn_attention=True
        )
        == expected_overlap
    )
    assert _get_cuda_graph_recompute_overlap(cuda_graph_modules, ["gdn"]) == []


def test_router_overlap_excludes_eager_shared_experts():
    """Shared experts outside the router graph must not produce a false warning."""
    assert _get_cuda_graph_recompute_overlap(
        [CudaGraphModule.moe_router],
        _RECOMPUTE_MODULES,
        router_captures_shared_experts=False,
    ) == ["moe"]



def _te_planned_double_buffer_args(cuda_graph_warmup_steps=3, **overrides):
    values = dict(
        cuda_graph_impl="transformer_engine",
        use_megatron_fsdp=True,
        cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        fp8_recipe=None,
        fp8_param_gather=False,
        megatron_fsdp_enable_fine_grained_param_gather=False,
        overlap_moe_expert_parallel_comm=False,
        nccl_ub=False,
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        fsdp_double_buffer=True,
        fsdp_db_use_persist_buf_on_alloc_fail=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _te_planned_double_buffer_config(**overrides):
    values = dict(
        fp8_param_gather=False,
        megatron_fsdp_enable_fine_grained_param_gather=False,
        megatron_fsdp_cuda_graph_mode=True,
        megatron_fsdp_use_planned_double_buffer=True,
        fsdp_double_buffer=True,
        fsdp_db_use_persist_buf_on_alloc_fail=False,
        nccl_ub=False,
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_param_gather=True,
        overlap_grad_reduce=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _local_double_buffer_config(**overrides):
    values = vars(
        _te_planned_double_buffer_config(
            megatron_fsdp_use_planned_double_buffer=False,
            fsdp_db_use_persist_buf_on_alloc_fail=True,
        )
    ).copy()
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"megatron_fsdp_cuda_graph_mode": False},
            "megatron_fsdp_cuda_graph_mode=True",
            id="graph-mode",
        ),
        pytest.param(
            {"fsdp_double_buffer": False}, "fsdp_double_buffer=True", id="double-buffer"
        ),
        pytest.param(
            {"fsdp_db_use_persist_buf_on_alloc_fail": False},
            "fsdp_db_use_persist_buf_on_alloc_fail=True",
            id="persistent-spill",
        ),
    ],
)
def test_adapter_rejects_unsafe_local_cuda_graph_buffers(overrides, message):
    config = SimpleNamespace(cuda_graph_impl="local")

    with pytest.raises(ValueError, match=message):
        _validate_cuda_graph_config(config, _local_double_buffer_config(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [{}, {"fsdp_double_buffer": False, "nccl_ub": True}],
)
def test_adapter_accepts_safe_local_cuda_graph_buffers(overrides):
    config = SimpleNamespace(cuda_graph_impl="local")

    _validate_cuda_graph_config(config, _local_double_buffer_config(**overrides))


def test_te_planned_double_buffer_requires_eager_warmup():
    """Lifetime coloring cannot be frozen without an eager observation step."""
    args = _te_planned_double_buffer_args(cuda_graph_warmup_steps=0)

    with pytest.raises(ValueError, match="cuda-graph-warmup-steps >= 1"):
        _validate_megatron_fsdp_cuda_graph_buffers(args)


@pytest.mark.parametrize("cuda_graph_warmup_steps", [1, 3])
def test_te_planned_double_buffer_accepts_positive_warmup(cuda_graph_warmup_steps):
    """One or more eager steps provide a lifetime trace for planned coloring."""
    args = _te_planned_double_buffer_args(cuda_graph_warmup_steps=cuda_graph_warmup_steps)

    _validate_megatron_fsdp_cuda_graph_buffers(args)


@pytest.mark.parametrize("disabled_overlap", ["overlap_param_gather", "overlap_grad_reduce"])
def test_te_planned_double_buffer_requires_overlapped_fsdp_communication(disabled_overlap):
    """A frozen two-bank plan cannot hold synchronous whole-model residency."""
    args = _te_planned_double_buffer_args(**{disabled_overlap: False})

    with pytest.raises(ValueError, match="requires --overlap-param-gather"):
        _validate_megatron_fsdp_cuda_graph_buffers(args)


def test_adapter_rejects_planned_double_buffer_without_eager_warmup():
    """Programmatic callers cannot bypass lifetime observation."""
    config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_warmup_steps=0,
        fp8_recipe=None,
        overlap_moe_expert_parallel_comm=False,
    )

    with pytest.raises(ValueError, match="cuda_graph_warmup_steps >= 1"):
        _validate_cuda_graph_config(config, _te_planned_double_buffer_config())


@pytest.mark.parametrize("cuda_graph_warmup_steps", [1, 3])
def test_adapter_accepts_positive_eager_warmup(cuda_graph_warmup_steps):
    """Direct API validation accepts an observed planned lifetime."""
    config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        fp8_recipe=None,
        overlap_moe_expert_parallel_comm=False,
    )

    _validate_cuda_graph_config(config, _te_planned_double_buffer_config())


@pytest.mark.parametrize("disabled_overlap", ["overlap_param_gather", "overlap_grad_reduce"])
def test_adapter_requires_overlapped_fsdp_communication(disabled_overlap):
    """Programmatic callers must preserve the same planned residency schedule."""
    config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_warmup_steps=3,
        fp8_recipe=None,
        overlap_moe_expert_parallel_comm=False,
    )
    ddp_config = _te_planned_double_buffer_config(**{disabled_overlap: False})

    with pytest.raises(ValueError, match="requires overlap_param_gather=True"):
        _validate_cuda_graph_config(config, ddp_config)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"fp8_recipe": "mxfp8", "fp8_param_gather": True}, id="mxfp8-param-gather"),
        pytest.param(
            {"megatron_fsdp_enable_fine_grained_param_gather": True},
            id="explicit-fine-grained-gather",
        ),
        pytest.param(
            {"overlap_moe_expert_parallel_comm": True}, id="ep-overlap-fine-grained-gather"
        ),
    ],
)
def test_cli_rejects_planned_double_buffer_with_fine_grained_gather(overrides):
    """CLI validation rejects every source of fine-grained FSDP hooks."""
    args = _te_planned_double_buffer_args(**overrides)

    with pytest.raises(ValueError, match="does not support fine-grained parameter gather hooks"):
        _validate_megatron_fsdp_cuda_graph_buffers(args)


@pytest.mark.parametrize(
    ("fp8_recipe", "overlap_moe_expert_parallel_comm", "ddp_overrides"),
    [
        pytest.param(
            "mxfp8",
            False,
            {"fp8_param_gather": True},
            id="mxfp8-param-gather",
        ),
        pytest.param(
            None,
            False,
            {"megatron_fsdp_enable_fine_grained_param_gather": True},
            id="explicit-fine-grained-gather",
        ),
        pytest.param(None, True, {}, id="ep-overlap-fine-grained-gather"),
    ],
)
def test_adapter_rejects_planned_double_buffer_with_fine_grained_gather(
    fp8_recipe, overlap_moe_expert_parallel_comm, ddp_overrides
):
    """Programmatic validation rejects every source of fine-grained FSDP hooks."""
    config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_warmup_steps=3,
        fp8_recipe=fp8_recipe,
        overlap_moe_expert_parallel_comm=overlap_moe_expert_parallel_comm,
    )
    ddp_config = _te_planned_double_buffer_config(**ddp_overrides)

    with pytest.raises(ValueError, match="does not support fine-grained parameter gather hooks"):
        _validate_cuda_graph_config(config, ddp_config)


class _RecordingDP:
    """Minimal DDP stand-in that records the original module."""

    def __init__(self, *, module, **kwargs):
        self.module = module
        self.allocator_namespace = _get_allocator_namespace(module)


def test_model_chunks_receive_distinct_namespace_labels():
    """VPP chunks receive distinct human-readable labels before unique PGB IDs."""
    chunks = [nn.Linear(2, 2), nn.Linear(2, 2)]
    ddp_config = SimpleNamespace(bucket_size=None, use_distributed_optimizer=False)

    wrapped = wrap_model_chunks_with_ddp(chunks, object(), ddp_config, DP=_RecordingDP)

    assert [chunk.module._megatron_fsdp_buffer_namespace for chunk in wrapped] == [
        "model_chunk_0",
        "model_chunk_1",
    ]


def test_separate_single_chunk_wraps_get_unique_allocator_namespaces():
    """Independent model groups cannot collide even with the same chunk label."""
    ddp_config = SimpleNamespace(bucket_size=None, use_distributed_optimizer=False)

    first = wrap_model_chunks_with_ddp(
        [nn.Linear(2, 2)], object(), ddp_config, DP=_RecordingDP
    )[0]
    second = wrap_model_chunks_with_ddp(
        [nn.Linear(2, 2)], object(), ddp_config, DP=_RecordingDP
    )[0]

    assert first.module._megatron_fsdp_buffer_namespace == "model_chunk_0"
    assert second.module._megatron_fsdp_buffer_namespace == "model_chunk_0"
    assert first.allocator_namespace != second.allocator_namespace
    assert first.allocator_namespace.startswith("model_chunk_0_param_and_grad_buffer_")
    assert second.allocator_namespace.startswith("model_chunk_0_param_and_grad_buffer_")


def test_wrap_preserves_caller_provided_namespace_label():
    """Upcycling callers may provide a more descriptive chunk label."""
    chunk = nn.Linear(2, 2)
    chunk._megatron_fsdp_buffer_namespace = "upcycling_source"
    ddp_config = SimpleNamespace(bucket_size=None, use_distributed_optimizer=False)

    wrapped = wrap_model_chunks_with_ddp(
        [chunk], object(), ddp_config, DP=_RecordingDP
    )[0]

    assert wrapped.module._megatron_fsdp_buffer_namespace == "upcycling_source"
    assert wrapped.allocator_namespace.startswith("upcycling_source_param_and_grad_buffer_")
