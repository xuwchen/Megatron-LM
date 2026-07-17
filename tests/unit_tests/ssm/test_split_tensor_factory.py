# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.ssm.gated_delta_net import (
    _split_tensor_factory as gated_delta_split_tensor_factory,
)
from megatron.core.ssm.mamba_mixer import _split_tensor_factory as mamba_split_tensor_factory


@pytest.mark.parametrize(
    "factory_fn",
    [gated_delta_split_tensor_factory, mamba_split_tensor_factory],
    ids=["gated_delta_net", "mamba_mixer"],
)
@pytest.mark.internal
def test_ssm_split_tensor_factory_oom_is_handled(factory_fn, caplog):
    original_sh_ten = ShardedTensor.from_rank_offsets(
        'a', torch.arange(12, dtype=torch.float32).reshape(6, 2), (0, 0, 1)
    )
    factory = factory_fn(original_sh_ten, [2, 4], ['x', 'B'], 0)
    sub_state_dict = [torch.ones((2, 2), dtype=torch.float32), torch.full((4, 2), 2.0)]

    real_cat = torch.cat
    call_count = 0

    def fake_cat(tensors, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError('mock oom')
        return real_cat(tensors, *args, **kwargs)

    with (
        mock.patch('torch.cat', side_effect=fake_cat),
        mock.patch('gc.collect') as collect_mock,
        mock.patch('torch.cuda.empty_cache') as empty_cache_mock,
        caplog.at_level(logging.WARNING),
    ):
        merged = factory.merge_fn(sub_state_dict)

    assert torch.equal(merged, real_cat(sub_state_dict))
    assert merged.device.type == 'cpu'
    assert call_count == 2
    collect_mock.assert_called_once()
    empty_cache_mock.assert_called_once()
    assert "CUDA OutOfMemoryError encountered during tensors merging" in caplog.text


@pytest.mark.internal
@pytest.mark.parametrize(
    ("dp_rank", "expected"),
    [
        (1, [("query", 504, 1544), ("key", 1040, 0)]),
        (2, [("key", 1008, 1040), ("value", 536, 0)]),
        (7, [("z", 1480, 2616), ("beta", 32, 0), ("alpha", 32, 0)]),
    ],
)
def test_gated_delta_split_tensor_factory_handles_cross_section_fsdp_shards(
    dp_rank, expected
):
    split_sections = [2048, 2048, 4096, 4096, 32, 32]
    split_names = ["query", "key", "value", "z", "beta", "alpha"]
    local_size = sum(split_sections) // 8
    local_start = dp_rank * local_size
    data = torch.arange(local_start, local_start + local_size, dtype=torch.float32).view(-1, 1)
    sharded_tensor = ShardedTensor(
        key="in_proj.weight",
        data=data,
        dtype=data.dtype,
        local_shape=tuple(data.shape),
        global_shape=(sum(split_sections), 1),
        global_offset=(local_start, 0),
        axis_fragmentations=(8, 1),
    )

    factory = gated_delta_split_tensor_factory(
        sharded_tensor, split_sections, split_names, split_dim=0
    )
    chunks = factory.build()

    actual = [
        (
            chunk.key.rsplit(".", 1)[-1],
            chunk.local_shape[0],
            chunk.global_offset[0],
        )
        for chunk in chunks
    ]
    assert actual == expected
    assert all(chunk.axis_fragmentations is None for chunk in chunks)
    assert torch.equal(factory.merge_fn([chunk.data for chunk in chunks]), data)
    assert all(
        chunk.data.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()
        for chunk in chunks
    )


@pytest.mark.internal
def test_gated_delta_split_tensor_factory_preserves_tp_offset_with_fsdp():
    # TP rank 1, DP rank 1 of a TP2/DP3 layout. The physical shard covers
    # rows [2, 4) in the TP-local fused tensor and maps into section "x".
    data = torch.arange(2, dtype=torch.float32).view(-1, 1)
    sharded_tensor = ShardedTensor(
        key="a",
        data=data,
        dtype=data.dtype,
        local_shape=(2, 1),
        global_shape=(12, 1),
        global_offset=(8, 0),
        axis_fragmentations=(6, 1),
    )

    factory = gated_delta_split_tensor_factory(sharded_tensor, [4, 2], ["x", "B"], 0)
    chunks = factory.build()

    assert len(chunks) == 1
    assert chunks[0].key == "a.x"
    assert chunks[0].global_shape == (8, 1)
    assert chunks[0].global_offset == (6, 0)
    assert torch.equal(factory.merge_fn([chunks[0].data]), data)
