# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch

import megatron.core.utils as utils


class FakeDTensor:
    """Minimal DTensor surface used by checkpoint conversion helpers."""

    def __init__(self, local_tensor):
        self._local_tensor = local_tensor
        self.device_mesh = SimpleNamespace(ndim=1, shape=(1,))
        self.placements = (object(),)
        self.shape = local_tensor.shape


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
        local_tensor,
        "linear_fc1.weight",
        tp_axis=0,
        tp_group=object(),
        dp_cp_group=object(),
    )
    assert not hasattr(plain_sharded_tensor, "is_data_parallel_fully_shard")


def test_non_tp_dtensor_checkpoint_shard_is_marked_fully_sharded(monkeypatch):
    monkeypatch.setattr(utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(utils, "DTensor", FakeDTensor)
    monkeypatch.setattr(utils, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(utils, "get_pg_size", lambda group: 1)

    local_tensor = torch.arange(4, dtype=torch.float32)
    sharded_tensor = utils.make_sharded_tensor_for_checkpoint(
        FakeDTensor(local_tensor),
        "linear_fc1.bias",
        tp_group=object(),
        dp_cp_group=object(),
    )

    assert sharded_tensor.data is local_tensor
    assert sharded_tensor.replica_id == (0, 0, 0)
    assert sharded_tensor.is_data_parallel_fully_shard is True
