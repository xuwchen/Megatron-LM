# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory


@pytest.mark.internal
@pytest.mark.parametrize("singleton_local_shards", [False, True])
def test_swiglu_factory_splits_cross_section_fsdp_shard(singleton_local_shards):
    # DP rank 1 of a DP3 layout owns fused rows [4, 8), crossing the
    # SwiGLU w/v boundary at row 6.
    data = torch.arange(8, dtype=torch.float32).view(4, 2)
    sharded_tensor = ShardedTensor(
        key="linear_fc1.weight",
        data=data,
        dtype=data.dtype,
        local_shape=(4, 2),
        global_shape=(12, 2),
        global_offset=(4, 0),
        axis_fragmentations=(3, 1),
        replica_id=(0, 0, 1),
    )
    sharded_tensor.is_data_parallel_fully_shard = True

    factory = apply_swiglu_sharded_factory(
        sharded_tensor, (), singleton_local_shards=singleton_local_shards, tp_size=1
    )
    chunks = factory.build()

    assert factory.is_data_parallel_fully_shard is True
    assert factory.replica_id == (0, 0, 1)
    assert all(chunk.replica_id == (0, 0, 1) for chunk in chunks)
    assert [chunk.local_shape for chunk in chunks] == [(2, 2), (2, 2)]
    assert all(chunk.axis_fragmentations is None for chunk in chunks)
    if singleton_local_shards:
        assert [chunk.key for chunk in chunks] == ["linear_fc1.weight_w", "linear_fc1.weight_v"]
        assert [chunk.global_shape for chunk in chunks] == [(6, 2), (6, 2)]
        assert [chunk.global_offset for chunk in chunks] == [(4, 0), (0, 0)]
    else:
        assert [chunk.key for chunk in chunks] == ["linear_fc1.weight", "linear_fc1.weight"]
        assert [chunk.global_shape for chunk in chunks] == [(12, 2), (12, 2)]
        assert [chunk.global_offset for chunk in chunks] == [(4, 0), (6, 0)]

    assert torch.equal(factory.merge_fn([chunk.data for chunk in chunks]), data)
    assert all(
        chunk.data.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()
        for chunk in chunks
    )

    optimizer_data = torch.full_like(data, 7)
    optimizer_chunks = factory.build_fn(
        "optimizer.linear_fc1.weight", optimizer_data, replica_id=0, flattened_range=None
    )
    assert torch.equal(factory.merge_fn([chunk.data for chunk in optimizer_chunks]), optimizer_data)


@pytest.mark.internal
def test_swiglu_factory_preserves_expert_prepend_axis_for_fsdp_shard():
    data = torch.arange(6, 8, dtype=torch.float32).view(2, 1)
    sharded_tensor = ShardedTensor(
        key="linear_fc1.weight3",
        data=data,
        dtype=data.dtype,
        local_shape=(2, 1),
        global_shape=(8, 1),
        global_offset=(6, 0),
        axis_fragmentations=(4, 1),
    )

    factory = apply_swiglu_sharded_factory(
        sharded_tensor, ((0, 3, 16),), singleton_local_shards=False, tp_size=1
    )
    chunks = factory.build()

    assert len(chunks) == 1
    assert chunks[0].key == "linear_fc1.weight3"
    assert chunks[0].local_shape == (2, 1)
    assert chunks[0].global_shape == (16, 8, 1)
    assert chunks[0].global_offset == (3, 6, 0)
    assert chunks[0].prepend_axis_num == 1
    assert torch.equal(factory.merge_fn([chunks[0].data]), data)
