# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestTinyFSDPParamCheckpoint:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_model_and_optimizer_roundtrip(self, tmp_path_dist_ckpt):
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            pytest.skip("This test requires at least two FSDP ranks.")

        mesh = DeviceMesh.from_group(
            parallel_state.get_data_parallel_group(with_context_parallel=True), "cuda"
        )
        model = torch.nn.Linear(8, 1, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.copy_(torch.arange(8, device="cuda").reshape(1, 8))
        expected_full_model = model.weight.detach().clone()
        fully_shard(model, mesh=mesh)
        param = next(model.parameters())

        model_factory = make_sharded_tensor_for_checkpoint(param, "weight")
        assert isinstance(model_factory, ShardedTensorFactory)
        assert model_factory.data is param._local_tensor
        param_map = get_param_id_to_sharded_param_map({"weight": model_factory}, [param])
        assert set(param_map) == {0}
        assert param_map[0] is model_factory

        built_model = model_factory.build()
        assert isinstance(built_model, ShardedTensor)
        assert built_model.local_shape == tuple(expected_full_model.shape)
        assert built_model.global_shape == tuple(expected_full_model.shape)
        assert built_model.replica_id == (0, 0, torch.distributed.get_rank())
        torch.testing.assert_close(built_model.data, expected_full_model)

        optimizer_local = torch.full_like(param._local_tensor, 3.0, dtype=torch.float32)
        optimizer_factory = make_sharded_optimizer_tensor(
            model_factory, optimizer_local, prefix="optimizer.state.exp_avg"
        )
        assert isinstance(optimizer_factory, ShardedTensorFactory)
        assert optimizer_factory.data is optimizer_local
        built_optimizer = optimizer_factory.build()
        assert isinstance(built_optimizer, ShardedTensor)
        assert built_optimizer.local_shape == tuple(expected_full_model.shape)
        torch.testing.assert_close(
            built_optimizer.data, torch.full_like(expected_full_model, 3.0, dtype=torch.float32)
        )

        expected_local_model = param._local_tensor.detach().clone()
        expected_local_optimizer = optimizer_local.detach().clone()

        with TempNamedDir(
            tmp_path_dist_ckpt / "test_tiny_fsdp_param_roundtrip", sync=True
        ) as ckpt_dir:
            save({"model": model_factory, "optimizer": optimizer_factory}, ckpt_dir)
            param._local_tensor.zero_()
            loaded = load(
                {
                    "model": make_sharded_tensor_for_checkpoint(param, "weight"),
                    "optimizer": make_sharded_optimizer_tensor(
                        make_sharded_tensor_for_checkpoint(param, "weight"),
                        torch.zeros_like(optimizer_local),
                        prefix="optimizer.state.exp_avg",
                    ),
                },
                ckpt_dir,
            )

        assert loaded["model"].shape == param._local_tensor.shape
        assert loaded["optimizer"].shape == optimizer_local.shape
        torch.testing.assert_close(loaded["model"], expected_local_model)
        torch.testing.assert_close(loaded["optimizer"], expected_local_optimizer)
