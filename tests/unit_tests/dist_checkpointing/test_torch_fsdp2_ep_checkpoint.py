# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

import pytest
import torch

from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
)
from megatron.core.distributed.torch_fully_sharded_data_parallel_config import (
    TorchFullyShardedDataParallelConfig,
)
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from megatron.core.utils import is_torch_min_version, make_sharded_tensor_for_checkpoint
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class _TinyDenseExpertModel(MegatronModule):
    """Small model with globally indexed experts and separate dense/expert optimizers."""

    def __init__(self, config, pg_collection, num_global_experts=2, init_shift=0):
        super().__init__(config)
        self.tp_group = pg_collection.tp
        self.ep_group = pg_collection.ep
        self.num_global_experts = num_global_experts
        num_local_experts = num_global_experts // self.ep_group.size()

        self.dense_weight = torch.nn.Parameter(
            torch.empty(8, 8, device="cuda", dtype=torch.bfloat16)
        )
        self.local_experts = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(8, 8, device="cuda", dtype=torch.bfloat16))
                for _ in range(num_local_experts)
            ]
        )
        for expert_weight in self.local_experts:
            expert_weight.allreduce = False

        first_global_expert = self.ep_group.rank() * num_local_experts
        with torch.no_grad():
            dense_values = torch.arange(64, device="cuda", dtype=torch.float32).view(8, 8)
            self.dense_weight.copy_((dense_values + init_shift) / 100)
            for local_idx, expert_weight in enumerate(self.local_experts):
                global_idx = first_global_expert + local_idx
                expert_weight.copy_(
                    (dense_values + 64 * (global_idx + 1) + init_shift) / 100
                )

    def forward(self, inputs):
        output = inputs @ self.dense_weight
        for expert_weight in self.local_experts:
            output = output + inputs @ expert_weight
        return output

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Map rank-local expert parameters onto one global expert axis."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state_dict = {
            f"{prefix}dense_weight": make_sharded_tensor_for_checkpoint(
                self.dense_weight,
                f"{prefix}dense_weight",
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        }

        num_local_experts = len(self.local_experts)
        first_global_expert = self.ep_group.rank() * num_local_experts
        for local_idx, expert_weight in enumerate(self.local_experts):
            global_idx = first_global_expert + local_idx
            expert_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), global_idx, self.num_global_experts),
            )
            state_dict[f"{prefix}local_experts.{local_idx}"] = (
                make_sharded_tensor_for_checkpoint(
                    expert_weight,
                    f"{prefix}experts.weight",
                    prepend_offsets=expert_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
            )
        return state_dict


def _build_model_and_optimizer(expert_parallel_size, *, init_shift):
    Utils.initialize_model_parallel(
        1,
        1,
        expert_model_parallel_size=expert_parallel_size,
        num_distributed_optimizer_instances=2,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        bf16=True,
    )
    config.expert_model_parallel_size = expert_parallel_size
    ddp_config = TorchFullyShardedDataParallelConfig(
        num_distributed_optimizer_instances=2,
        reduce_scatter_unused_params=True,
    )
    model = TorchFullyShardedDataParallel(
        config,
        ddp_config,
        _TinyDenseExpertModel(config, pg_collection, init_shift=init_shift),
        sub_modules_to_wrap=set(),
        pg_collection=pg_collection,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-2,
            weight_decay=0.0,
            clip_grad=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
        ),
        [model],
        use_gloo_process_groups=False,
        pg_collection=pg_collection,
    )
    assert isinstance(optimizer, ChainedOptimizer)
    assert len(optimizer.chained_optimizers) == 2
    return model, optimizer, pg_collection


def _run_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(16, device="cuda", dtype=torch.bfloat16).view(2, 8) / 16
    model(inputs).float().square().mean().backward()
    success, _, _ = optimizer.step()
    assert success
    for sub_optimizer in optimizer.chained_optimizers:
        assert sub_optimizer.optimizer.state
        for state in sub_optimizer.optimizer.state.values():
            assert {"exp_avg", "exp_avg_sq"} <= state.keys()


def _sharded_training_state(model, optimizer, pg_collection, *, is_loading=False):
    metadata = {
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": pg_collection.dp_cp,
        "singleton_local_shards": False,
    }
    state_dict = {"model": model.sharded_state_dict(metadata=metadata)}
    state_dict["optimizer"] = optimizer.sharded_state_dict(
        state_dict, metadata=metadata, is_loading=is_loading
    )
    return state_dict


@pytest.mark.parametrize(("source_ep_size", "destination_ep_size"), [(2, 2), (1, 2), (2, 1)])
def test_fsdp2_ep_hsdp_model_and_optimizer_roundtrip(
    tmp_path_dist_ckpt, source_ep_size, destination_ep_size
):
    """Reshard distinct dense/expert HSDP meshes and BF16 Adam main state."""
    if Utils.world_size != 8:
        pytest.skip("This FSDP2 EP/HSDP checkpoint test requires exactly eight ranks.")
    if not is_torch_min_version("2.13.0"):
        pytest.skip("Per-parameter FSDP2 meshes require PyTorch >= 2.13.")

    Utils.initialize_distributed()
    topology = f"ep{source_ep_size}_to_ep{destination_ep_size}"
    with TempNamedDir(tmp_path_dist_ckpt / f"fsdp2_ep_hsdp_A_{topology}", sync=True) as ckpt_dir_a:
        with TempNamedDir(
            tmp_path_dist_ckpt / f"fsdp2_ep_hsdp_B_{topology}", sync=True
        ) as ckpt_dir_b:
            try:
                model, optimizer, pg_collection = _build_model_and_optimizer(
                    source_ep_size, init_shift=0
                )
                _run_step(model, optimizer)
                save(_sharded_training_state(model, optimizer, pg_collection), ckpt_dir_a)
                model = optimizer = pg_collection = None
                gc.collect()
                Utils.destroy_model_parallel()

                model, optimizer, pg_collection = _build_model_and_optimizer(
                    destination_ep_size, init_shift=10_000
                )
                dense_before_load = model.module.dense_weight._local_tensor.detach().clone()
                load_template = _sharded_training_state(
                    model, optimizer, pg_collection, is_loading=True
                )
                loaded_state = load(load_template, ckpt_dir_a)
                optimizer.load_common_state_dict(loaded_state["optimizer"])
                assert not torch.equal(
                    dense_before_load, model.module.dense_weight._local_tensor
                )
                save(_sharded_training_state(model, optimizer, pg_collection), ckpt_dir_b)
                model = optimizer = pg_collection = load_template = loaded_state = None
                gc.collect()
                Utils.destroy_model_parallel()

                Utils.initialize_model_parallel(1, 1)
                plain_state_dict_a = load_plain_tensors(ckpt_dir_a)
                plain_state_dict_b = load_plain_tensors(ckpt_dir_b)
                expected_tensor_keys = {
                    "module.dense_weight",
                    "module.experts.weight",
                    "optimizer.state.exp_avg.module.dense_weight",
                    "optimizer.state.exp_avg.module.experts.weight",
                    "optimizer.state.exp_avg_sq.module.dense_weight",
                    "optimizer.state.exp_avg_sq.module.experts.weight",
                    "optimizer.state.fp32_param.module.dense_weight",
                    "optimizer.state.fp32_param.module.experts.weight",
                }
                assert expected_tensor_keys <= plain_state_dict_a.keys()
                assert not any("chained_" in key for key in plain_state_dict_a)
                diffs = diff(plain_state_dict_a, plain_state_dict_b)
                assert not any(map(bool, diffs)), diffs
            finally:
                Utils.destroy_model_parallel()


def test_fsdp2_ep_hsdp_resume_step_parity(tmp_path_dist_ckpt):
    """A resumed optimizer produces the same next step as uninterrupted training."""
    if Utils.world_size != 8:
        pytest.skip("This FSDP2 EP/HSDP checkpoint test requires exactly eight ranks.")
    if not is_torch_min_version("2.13.0"):
        pytest.skip("Per-parameter FSDP2 meshes require PyTorch >= 2.13.")

    Utils.initialize_distributed()
    with TempNamedDir(tmp_path_dist_ckpt / "fsdp2_resume_start", sync=True) as start_dir:
        with TempNamedDir(
            tmp_path_dist_ckpt / "fsdp2_resume_expected", sync=True
        ) as expected_dir:
            with TempNamedDir(
                tmp_path_dist_ckpt / "fsdp2_resume_actual", sync=True
            ) as actual_dir:
                try:
                    model, optimizer, pg_collection = _build_model_and_optimizer(
                        2, init_shift=0
                    )
                    _run_step(model, optimizer)
                    save(_sharded_training_state(model, optimizer, pg_collection), start_dir)
                    _run_step(model, optimizer)
                    save(_sharded_training_state(model, optimizer, pg_collection), expected_dir)
                    model = optimizer = pg_collection = None
                    gc.collect()
                    Utils.destroy_model_parallel()

                    model, optimizer, pg_collection = _build_model_and_optimizer(
                        2, init_shift=10_000
                    )
                    load_template = _sharded_training_state(
                        model, optimizer, pg_collection, is_loading=True
                    )
                    loaded_state = load(load_template, start_dir)
                    optimizer.load_common_state_dict(loaded_state["optimizer"])
                    _run_step(model, optimizer)
                    save(_sharded_training_state(model, optimizer, pg_collection), actual_dir)
                    model = optimizer = pg_collection = load_template = loaded_state = None
                    gc.collect()
                    Utils.destroy_model_parallel()

                    Utils.initialize_model_parallel(1, 1)
                    expected_state = load_plain_tensors(expected_dir)
                    actual_state = load_plain_tensors(actual_dir)
                    diffs = diff(expected_state, actual_state)
                    assert not any(map(bool, diffs)), diffs
                finally:
                    Utils.destroy_model_parallel()
