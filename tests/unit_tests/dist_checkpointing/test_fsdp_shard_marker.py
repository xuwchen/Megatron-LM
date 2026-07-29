# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

import megatron.core.utils as utils


class FakeMesh:
    """Minimal DeviceMesh surface used by checkpoint conversion helpers."""

    def __init__(self, shape, coordinate):
        self.ndim = len(shape)
        self.shape = shape
        self._coordinate = coordinate

    def get_coordinate(self):
        return self._coordinate


class FakeDTensor:
    """Minimal DTensor surface used by checkpoint conversion helpers."""

    def __init__(
        self, local_tensor, *, mesh_shape=(1,), coordinate=(0,), placements=None, global_shape=None
    ):
        self._local_tensor = local_tensor
        self.device_mesh = FakeMesh(mesh_shape, coordinate)
        self.placements = placements or (utils.Shard(0),)
        self.shape = global_shape or local_tensor.shape


def test_tp_dtensor_checkpoint_shard_is_marked_fully_sharded(monkeypatch):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)
    monkeypatch.setattr(utils, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(utils, "get_pg_size", lambda group: 1)

    local_tensor = torch.arange(8, dtype=torch.float32).view(4, 2)
    sharded_tensor = utils.make_tp_sharded_tensor_for_checkpoint(
        FakeDTensor(local_tensor),
        "linear_fc1.weight",
        tp_axis=0,
        tp_group=object(),
        dp_cp_group=object(),
    )

    assert sharded_tensor.data is local_tensor
    assert sharded_tensor.replica_id == (0, 0, 0)
    assert sharded_tensor.is_data_parallel_fully_shard is True

    plain_sharded_tensor = utils.make_tp_sharded_tensor_for_checkpoint(
        local_tensor, "linear_fc1.weight", tp_axis=0, tp_group=object(), dp_cp_group=object()
    )
    assert not hasattr(plain_sharded_tensor, "is_data_parallel_fully_shard")


def test_non_tp_dtensor_checkpoint_shard_is_marked_fully_sharded(monkeypatch):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)
    monkeypatch.setattr(utils, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(utils, "get_pg_size", lambda group: 1)

    local_tensor = torch.arange(4, dtype=torch.float32)
    sharded_tensor = utils.make_sharded_tensor_for_checkpoint(
        FakeDTensor(local_tensor), "linear_fc1.bias", tp_group=object(), dp_cp_group=object()
    )

    assert sharded_tensor.data is local_tensor
    assert sharded_tensor.replica_id == (0, 0, 0)
    assert sharded_tensor.is_data_parallel_fully_shard is True


def test_hsdp_checkpoint_layout_comes_from_dtensor_mesh(monkeypatch):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)
    monkeypatch.setattr(utils, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(utils, "get_pg_size", lambda group: 8)

    local_tensor = torch.arange(4, dtype=torch.float32).view(2, 2)
    tensor = FakeDTensor(
        local_tensor,
        mesh_shape=(2, 4),
        coordinate=(1, 2),
        placements=(utils.Replicate(), utils.Shard(0)),
        global_shape=(8, 2),
    )
    sharded_tensor = utils.make_sharded_tensor_for_checkpoint(
        tensor, "dense.weight", tp_group=object(), dp_cp_group=object()
    )

    assert sharded_tensor.data is local_tensor
    assert sharded_tensor.global_shape == (8, 2)
    assert sharded_tensor.global_offset == (4, 0)
    assert sharded_tensor.axis_fragmentations == (4, 1)
    assert sharded_tensor.replica_id == (0, 0, 1)


def test_expert_hsdp_uses_its_smaller_dtensor_mesh(monkeypatch):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)
    monkeypatch.setattr(utils, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(utils, "get_pg_size", lambda group: 8)

    local_tensor = torch.arange(8, dtype=torch.float32).view(4, 2)
    tensor = FakeDTensor(
        local_tensor,
        mesh_shape=(2, 2),
        coordinate=(1, 1),
        placements=(utils.Replicate(), utils.Shard(0)),
        global_shape=(8, 2),
    )
    sharded_tensor = utils.make_sharded_tensor_for_checkpoint(
        tensor, "experts.weight", tp_group=object(), dp_cp_group=object()
    )

    assert sharded_tensor.global_shape == (8, 2)
    assert sharded_tensor.global_offset == (4, 0)
    assert sharded_tensor.axis_fragmentations == (2, 1)
    assert sharded_tensor.replica_id == (0, 0, 1)


def test_materialized_full_dtensor_uses_entire_mesh_as_replica_id(monkeypatch):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)

    tensor = FakeDTensor(
        torch.ones(1, 2),
        mesh_shape=(2, 4),
        coordinate=(1, 2),
        placements=(utils.Replicate(), utils.Shard(0)),
        global_shape=(1, 2),
    )

    assert utils._get_dtensor_checkpoint_full_tensor_replica_id(tensor) == 6


@pytest.mark.parametrize(
    "placements",
    [(utils.Replicate(),), (utils.Shard(0), utils.Shard(0)), (utils.Replicate(), utils.Shard(1))],
)
def test_unsupported_dtensor_checkpoint_layout_fails_fast(monkeypatch, placements):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)

    tensor = FakeDTensor(
        torch.ones(2, 2),
        mesh_shape=tuple(1 for _ in placements),
        coordinate=tuple(0 for _ in placements),
        placements=placements,
    )

    with pytest.raises(ValueError, match="FSDP2 checkpointing"):
        utils._get_dtensor_checkpoint_shard_info(tensor)


def test_uneven_dtensor_chunking_requires_full_tensor():
    # dim0=10 over a 4-way shard mesh chunks as [3, 3, 3, 1]: uneven, so the
    # regular-grid ShardedTensor path must be bypassed.
    uneven = FakeDTensor(torch.zeros(3, 2), mesh_shape=(4,), coordinate=(0,), global_shape=(10, 2))
    assert utils._dtensor_requires_full_tensor(uneven)

    # dim0=96 over a 64-way mesh chunks as 48x2 plus 16 EMPTY ranks even
    # though 64 > 96 is false; the divisibility term must catch it.
    empty_tail = FakeDTensor(
        torch.zeros(2, 2), mesh_shape=(64,), coordinate=(0,), global_shape=(96, 2)
    )
    assert utils._dtensor_requires_full_tensor(empty_tail)

    even = FakeDTensor(torch.zeros(2, 2), mesh_shape=(4,), coordinate=(0,), global_shape=(8, 2))
    assert not utils._dtensor_requires_full_tensor(even)


def test_tp_helper_rejects_empty_or_uneven_dtensor_shards(monkeypatch):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)
    monkeypatch.setattr(utils, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(utils, "get_pg_size", lambda group: 1)

    tiny = FakeDTensor(torch.ones(1, 2), mesh_shape=(4,), coordinate=(0,), global_shape=(1, 2))
    with pytest.raises(NotImplementedError, match="empty or uneven local chunks"):
        utils.make_tp_sharded_tensor_for_checkpoint(
            tiny, "linear_fc1.weight", tp_axis=0, tp_group=object(), dp_cp_group=object()
        )

    uneven = FakeDTensor(torch.ones(3, 2), mesh_shape=(4,), coordinate=(0,), global_shape=(10, 2))
    with pytest.raises(NotImplementedError, match="empty or uneven local chunks"):
        utils.make_tp_sharded_tensor_for_checkpoint(
            uneven, "linear_fc1.weight", tp_axis=0, tp_group=object(), dp_cp_group=object()
        )
