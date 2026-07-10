# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Megatron-FSDP CUDA graph argument validation."""

from types import SimpleNamespace

import pytest

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _validate_cuda_graph_config
from megatron.training.arguments import _validate_megatron_fsdp_cuda_graph_buffers


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
        fp8_recipe=fp8_recipe,
        overlap_moe_expert_parallel_comm=overlap_moe_expert_parallel_comm,
    )
    ddp_config = _te_planned_double_buffer_config(**ddp_overrides)

    with pytest.raises(ValueError, match="does not support fine-grained parameter gather hooks"):
        _validate_cuda_graph_config(config, ddp_config)
