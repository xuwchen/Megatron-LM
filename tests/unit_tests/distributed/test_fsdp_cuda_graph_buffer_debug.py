# Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.cuda_graph_buffer_debug import (
    GraphBufferKey,
    assert_cuda_graph_buffer_addresses,
    begin_cuda_graph_buffer_capture,
    begin_cuda_graph_buffer_replay,
    configure_cuda_graph_buffer_debug,
    finish_cuda_graph_buffer_capture,
    finish_cuda_graph_buffer_replay,
    record_cuda_graph_buffer_allocate,
    record_cuda_graph_buffer_free,
    reset_cuda_graph_buffer_debug_state,
)


def setup_function():
    reset_cuda_graph_buffer_debug_state()


def teardown_function():
    reset_cuda_graph_buffer_debug_state()


def _key():
    return GraphBufferKey(
        namespace="test",
        kind="weight_bucket",
        bucket_id=7,
        is_transpose=False,
        allocator_name="fsdp_params",
    )


def test_replay_address_match_passes():
    configure_cuda_graph_buffer_debug(assert_addresses=True)
    tensor = torch.empty(8, dtype=torch.float32)
    key = _key()

    begin_cuda_graph_buffer_capture("stage")
    record_cuda_graph_buffer_allocate(key, tensor, allocator_slot=(0, 1), source="test")
    finish_cuda_graph_buffer_capture("stage")

    begin_cuda_graph_buffer_replay("stage")
    record_cuda_graph_buffer_allocate(key, tensor, allocator_slot=(0, 1), source="test")
    assert_cuda_graph_buffer_addresses("stage")
    finish_cuda_graph_buffer_replay("stage")


def test_replay_address_mismatch_raises():
    configure_cuda_graph_buffer_debug(assert_addresses=True)
    key = _key()

    begin_cuda_graph_buffer_capture("stage")
    record_cuda_graph_buffer_allocate(key, torch.empty(8), allocator_slot=(0, 1), source="test")
    finish_cuda_graph_buffer_capture("stage")

    begin_cuda_graph_buffer_replay("stage")
    record_cuda_graph_buffer_allocate(key, torch.empty(8), allocator_slot=(1, 1), source="test")
    with pytest.raises(RuntimeError, match="CUDA graph buffer address mismatch"):
        assert_cuda_graph_buffer_addresses("stage")
    finish_cuda_graph_buffer_replay("stage")


def test_replay_freed_buffer_raises():
    configure_cuda_graph_buffer_debug(assert_addresses=True)
    key = _key()

    begin_cuda_graph_buffer_capture("stage")
    record_cuda_graph_buffer_allocate(key, torch.empty(8), allocator_slot=(0, 1), source="test")
    finish_cuda_graph_buffer_capture("stage")

    begin_cuda_graph_buffer_replay("stage")
    record_cuda_graph_buffer_free(key, allocator_slot=(0, 1), source="test")
    with pytest.raises(RuntimeError, match="buffer is currently freed"):
        assert_cuda_graph_buffer_addresses("stage")
    finish_cuda_graph_buffer_replay("stage")
