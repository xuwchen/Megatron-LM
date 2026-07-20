# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

import megatron.core.distributed.torch_fully_sharded_data_parallel as torch_fsdp2
import megatron.core.utils as core_utils
import megatron.training.utils as training_utils
from megatron.core import parallel_state
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
    _build_fsdp_device_mesh,
    _build_fsdp_wrap_plan,
    _clone_fsdp_output_views,
    _validate_fsdp_gradient_accumulation_mode,
)
from megatron.core.distributed.torch_fully_sharded_data_parallel_config import (
    TorchFullyShardedDataParallelConfig,
)
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal, is_torch_min_version
from tests.unit_tests.test_utilities import Utils


class DummyModel(MegatronModule):
    """Setup a few modules to test the FSDP2 constructor."""

    _fsdp_modules = [torch.nn.Linear]

    def __init__(self, config: TransformerConfig):
        """Initialize a dummy model with a few modules."""
        super().__init__(config)
        self.linear = torch.nn.Linear(2, 2)
        self.column_parallel_linear = ColumnParallelLinear(
            input_size=2, output_size=2, config=config, init_method=init_method_normal(0.02)
        )
        self.conv = torch.nn.Conv2d(2, 2, 1)


class _WrapLeaf(torch.nn.Linear):
    """Linear module matching multiple wrap types."""


class _WrapParent(torch.nn.Module):
    """Selected parent containing another selected module."""

    def __init__(self):
        super().__init__()
        self.leaf = _WrapLeaf(8, 8)
        self.bias = torch.nn.Parameter(torch.zeros(8))

    def forward(self, inputs):
        """Run the selected child and a parameter owned directly by this parent."""
        return self.leaf(inputs) + self.bias


class _NestedWrapRoot(torch.nn.Module):
    """Root containing nested FSDP wrap targets without shared aliases."""

    def __init__(self):
        super().__init__()
        self.parent = _WrapParent()

    def forward(self, inputs):
        """Run the nested parent."""
        return self.parent(inputs)


class _AutoReshardRoot(torch.nn.Module):
    """Root with its own parameter and one independently wrapped child."""

    def __init__(self):
        super().__init__()
        self.child = _WrapLeaf(8, 8)
        self.root_bias = torch.nn.Parameter(torch.zeros(8))

    def forward(self, inputs):
        """Run both the child and the parameter owned by the root FSDP group."""
        return self.child(inputs) + self.root_bias


class _ViewOutputRoot(torch.nn.Module):
    """Root module whose differentiable output aliases an intermediate tensor."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(8))

    def forward(self, inputs):
        """Return a view to exercise FSDP's output-hook safety requirement."""
        return (inputs @ self.weight).view(-1, 8)


class _MixedDenseExpertRoot(torch.nn.Module):
    """One FSDP unit with dense and routed-expert parameters."""

    def __init__(self):
        super().__init__()
        self.dense_weight = torch.nn.Parameter(torch.empty(8, 8))
        self.expert_weight = torch.nn.Parameter(torch.empty(8, 8))
        self.expert_weight.allreduce = False

    def forward(self, inputs, use_expert=True):
        """Optionally leave the expert parameter unused on a subset of ranks."""
        output = inputs @ self.dense_weight
        if use_expert:
            output = output + inputs @ self.expert_weight
        return output


@dataclass
class _ViewOutputContainer:
    """Dataclass output containing a repeated tensor reference."""

    primary: torch.Tensor
    aliases: tuple
    metadata: str


class _WrapRoot(torch.nn.Module):
    """Root matching a wrap type and sharing a nested child by identity."""

    def __init__(self):
        super().__init__()
        self.parent = _WrapParent()
        self.shared_leaf = self.parent.leaf


class _ConditionalParamBlock(torch.nn.Module):
    """Keep rank-divergent parameter usage inside one shared FSDP group."""

    def __init__(self):
        super().__init__()
        self.always = torch.nn.Linear(8, 8, bias=False)
        self.branch_a = torch.nn.Linear(8, 8, bias=False)
        self.branch_b = torch.nn.Linear(8, 8, bias=False)
        self.explicit_zero = torch.nn.Parameter(torch.ones(8))
        self.globally_unused = torch.nn.Parameter(torch.ones(8))
        self.frozen = torch.nn.Parameter(torch.ones(8), requires_grad=False)

    def forward(self, inputs, use_branch_a, use_branch_b):
        """Run the same FSDP module on every rank but select different parameters."""
        output = self.always(inputs) + 0.0 * self.explicit_zero
        if use_branch_a:
            output = output + self.branch_a(inputs)
        if use_branch_b:
            output = output + self.branch_b(inputs)
        return output


class _ConditionalParamRoot(torch.nn.Module):
    """Root module containing one conditionally used FSDP parameter group."""

    def __init__(self):
        super().__init__()
        self.block = _ConditionalParamBlock()
        self.root_scale = torch.nn.Parameter(torch.ones(8))
        self.root_branch = torch.nn.Parameter(torch.ones(8))
        self.root_globally_unused = torch.nn.Parameter(torch.ones(8))

    def forward(self, inputs, use_branch_a, use_branch_b, use_root_branch):
        """Delegate to the conditional block on every rank."""
        output = self.block(inputs, use_branch_a, use_branch_b) * self.root_scale
        if use_root_branch:
            output = output + self.root_branch
        return output


class _SyncControlSpy:
    """Record calls to the FSDP2 gradient synchronization controls."""

    def __init__(self):
        self.calls = []

    def set_is_last_backward(self, is_last_backward):
        self.calls.append(("last_backward", is_last_backward))

    def set_requires_gradient_sync(self, requires_gradient_sync, recurse):
        self.calls.append(("gradient_sync", requires_gradient_sync, recurse))


class _GradientSyncOnlySpy:
    """Emulate an older FSDP2 root without set_is_last_backward."""

    def __init__(self):
        self.calls = []

    def set_requires_gradient_sync(self, requires_gradient_sync, recurse):
        self.calls.append(("gradient_sync", requires_gradient_sync, recurse))


class _PartialSyncControlSpy:
    """Record partial reduce-scatter controls and reject classic synchronization."""

    def __init__(self):
        self.calls = []

    def set_is_last_backward(self, is_last_backward):
        self.calls.append(("last_backward", is_last_backward))

    def set_requires_all_reduce(self, requires_all_reduce, recurse):
        self.calls.append(("all_reduce", requires_all_reduce, recurse))

    def set_requires_gradient_sync(self, _requires_gradient_sync, _recurse):
        raise AssertionError("partial_reduce_scatter must keep reduce-scatter enabled")


def _get_partial_reduce_outputs(*fsdp_modules):
    """Return PyTorch's pending HSDP partial reductions for the given modules."""
    outputs = []
    for fsdp_module in fsdp_modules:
        fsdp_group = torch_fsdp2.fully_shard.state(fsdp_module)._fsdp_param_group
        if fsdp_group is not None:
            outputs.append(fsdp_group._partial_reduce_output)
    return outputs


_WRAP_TYPES = {_WrapRoot, _WrapParent, _WrapLeaf, torch.nn.Linear}


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


@pytest.fixture
def init_hsdp_model_parallel():
    """Initialize the exact 2x4 HSDP topology used by the 8-rank gate."""
    if Utils.world_size != 8:
        pytest.skip("This HSDP numerical test requires exactly 8 ranks.")
    Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=2)
    init_num_microbatches_calculator(
        rank=Utils.rank, global_batch_size=16, micro_batch_size=1, data_parallel_size=8
    )
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


@pytest.fixture
def init_ep_hsdp_model_parallel():
    """Initialize dense 2x4 and routed-expert 2x2 HSDP meshes on eight ranks."""
    if Utils.world_size != 8:
        pytest.skip("This EP HSDP numerical test requires exactly 8 ranks.")
    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=2, num_distributed_optimizer_instances=2
    )
    init_num_microbatches_calculator(
        rank=Utils.rank, global_batch_size=16, micro_batch_size=1, data_parallel_size=8
    )
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


def test_fsdp2_builds_hsdp_mesh_from_existing_process_groups(monkeypatch):
    """Reuse MCore's orthogonal groups without creating new communicators."""
    full_group = object()
    shard_group = object()
    replicate_group = object()
    fake_mesh = object()
    calls = []
    group_ranks = {full_group: list(range(8)), shard_group: [0, 1, 2, 3], replicate_group: [0, 4]}

    class _FakeDeviceMesh:
        @staticmethod
        def from_group(groups, device_type, mesh=None, mesh_dim_names=None):
            calls.append((groups, device_type, mesh, mesh_dim_names))
            return fake_mesh

    monkeypatch.setattr(torch_fsdp2, "DeviceMesh", _FakeDeviceMesh)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: group_ranks[group]
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    pg_collection = ProcessGroupCollection(
        dp_cp=full_group, intra_dp_cp=shard_group, inter_dist_opt=replicate_group
    )

    result = _build_fsdp_device_mesh(full_group, pg_collection, 2)

    assert result is fake_mesh
    assert calls == [
        (
            [replicate_group, shard_group],
            "cuda",
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            ("replicate", "shard"),
        )
    ]


def test_fsdp2_builds_expert_hsdp_mesh_from_existing_process_groups(monkeypatch):
    """Lay out sparse expert-DP ranks as two shard rows sharing dense replica columns."""
    full_group = object()
    shard_group = object()
    replicate_group = object()
    fake_mesh = object()
    calls = []
    group_ranks = {full_group: [0, 2, 4, 6], shard_group: [0, 2], replicate_group: [0, 4]}

    class _FakeDeviceMesh:
        @staticmethod
        def from_group(groups, device_type, mesh=None, mesh_dim_names=None):
            calls.append((groups, device_type, mesh, mesh_dim_names))
            return fake_mesh

    monkeypatch.setattr(torch_fsdp2, "DeviceMesh", _FakeDeviceMesh)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: group_ranks[group]
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    pg_collection = ProcessGroupCollection(
        expt_dp=full_group, intra_expt_dp=shard_group, inter_dist_opt=replicate_group
    )

    result = _build_fsdp_device_mesh(full_group, pg_collection, 2, expert_parallel=True)

    assert result is fake_mesh
    assert calls == [
        ([replicate_group, shard_group], "cuda", [[0, 2], [4, 6]], ("replicate", "shard"))
    ]


@pytest.mark.parametrize(
    ("shard_ranks", "replicate_ranks", "error"),
    [
        ([1, 2, 3, 4], [0, 4], "mesh row"),
        ([0, 1, 2, 3], [0, 5], "mesh column"),
        ([0, 1], [0, 2], "does not cover"),
    ],
)
def test_fsdp2_rejects_invalid_hsdp_topology(monkeypatch, shard_ranks, replicate_ranks, error):
    """Fail before fully_shard when MCore groups do not form the requested mesh."""
    full_group = object()
    shard_group = object()
    replicate_group = object()
    group_ranks = {
        full_group: list(range(8)),
        shard_group: shard_ranks,
        replicate_group: replicate_ranks,
    }
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: group_ranks[group]
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    pg_collection = ProcessGroupCollection(
        dp_cp=full_group, intra_dp_cp=shard_group, inter_dist_opt=replicate_group
    )

    with pytest.raises(ValueError, match=error):
        _build_fsdp_device_mesh(full_group, pg_collection, 2)


def test_dtensor_norm_group_uses_only_hsdp_shard_dimension(monkeypatch):
    """Do not count replicated HSDP shards twice in norms and clipping."""
    shard_group = object()
    requested_mesh_dims = []

    class _FakeMesh:
        ndim = 2

        @staticmethod
        def get_group(mesh_dim=None):
            requested_mesh_dims.append(mesh_dim)
            return shard_group

    class _FakeDTensor:
        device_mesh = _FakeMesh()
        placements = (Replicate(), Shard(0))

    monkeypatch.setattr(core_utils, "HAVE_DTENSOR", True)
    monkeypatch.setattr(core_utils, "DTensor", _FakeDTensor)

    result = core_utils.get_data_parallel_group_if_dtensor(_FakeDTensor())

    assert result is shard_group
    assert requested_mesh_dims == [1]


@pytest.mark.parametrize("force_all_reduce", [False, True])
def test_fsdp2_finish_grad_sync_accepts_force_all_reduce(monkeypatch, force_all_reduce):
    """Test compatibility with finalize_model_grads' keyword argument."""
    calls = []
    monkeypatch.setattr(
        _BaseDataParallel, "finish_grad_sync", lambda instance: calls.append(instance)
    )
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)

    fsdp_model.finish_grad_sync(force_all_reduce=force_all_reduce)

    assert calls == [fsdp_model]


@pytest.mark.parametrize("raise_in_context", [False, True])
def test_fsdp2_no_sync_restores_control_flags(raise_in_context):
    """Restore FSDP2 synchronization controls on normal and exceptional exits."""
    root_module = _SyncControlSpy()
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)
    object.__setattr__(fsdp_model, "module", root_module)
    object.__setattr__(fsdp_model, "_no_sync_depth", 0)

    if raise_in_context:
        with pytest.raises(RuntimeError, match="microbatch failure"):
            with fsdp_model.no_sync():
                root_module.calls.append("body")
                raise RuntimeError("microbatch failure")
    else:
        with fsdp_model.no_sync():
            root_module.calls.append("body")

    assert root_module.calls == [
        ("last_backward", False),
        ("gradient_sync", False, True),
        "body",
        ("gradient_sync", True, True),
        ("last_backward", True),
    ]
    assert fsdp_model._no_sync_depth == 0


def test_fsdp2_no_sync_is_reentrant():
    """Keep synchronization disabled until the outermost no_sync context exits."""
    root_module = _SyncControlSpy()
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)
    object.__setattr__(fsdp_model, "module", root_module)
    object.__setattr__(fsdp_model, "_no_sync_depth", 0)

    with fsdp_model.no_sync():
        root_module.calls.append("outer body")
        with fsdp_model.no_sync():
            root_module.calls.append("inner body")

    assert root_module.calls == [
        ("last_backward", False),
        ("gradient_sync", False, True),
        "outer body",
        "inner body",
        ("gradient_sync", True, True),
        ("last_backward", True),
    ]
    assert fsdp_model._no_sync_depth == 0


def test_fsdp2_no_sync_supports_older_fsdp2_controls():
    """Fall back to the synchronization control available since FSDP2 debuted."""
    root_module = _GradientSyncOnlySpy()
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)
    object.__setattr__(fsdp_model, "module", root_module)
    object.__setattr__(fsdp_model, "_no_sync_depth", 0)

    with fsdp_model.no_sync():
        root_module.calls.append("body")

    assert root_module.calls == [
        ("gradient_sync", False, True),
        "body",
        ("gradient_sync", True, True),
    ]
    assert fsdp_model._no_sync_depth == 0


@pytest.mark.parametrize("raise_in_context", [False, True])
def test_fsdp2_partial_reduce_scatter_restores_control_flags(raise_in_context):
    """Defer only the HSDP replica all-reduce and restore controls on every exit."""
    root_module = _PartialSyncControlSpy()
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)
    object.__setattr__(fsdp_model, "module", root_module)
    object.__setattr__(fsdp_model, "_no_sync_depth", 0)
    object.__setattr__(fsdp_model, "gradient_accumulation_mode", "partial_reduce_scatter")

    if raise_in_context:
        with pytest.raises(RuntimeError, match="microbatch failure"):
            with fsdp_model.no_sync():
                root_module.calls.append("body")
                raise RuntimeError("microbatch failure")
    else:
        with fsdp_model.no_sync():
            root_module.calls.append("body")

    assert root_module.calls == [
        ("last_backward", False),
        ("all_reduce", False, True),
        "body",
        ("all_reduce", True, True),
        ("last_backward", True),
    ]
    assert fsdp_model._no_sync_depth == 0


def test_fsdp2_partial_reduce_scatter_no_sync_is_reentrant():
    """Change partial-reduction controls only for the outermost context."""
    root_module = _PartialSyncControlSpy()
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)
    object.__setattr__(fsdp_model, "module", root_module)
    object.__setattr__(fsdp_model, "_no_sync_depth", 0)
    object.__setattr__(fsdp_model, "gradient_accumulation_mode", "partial_reduce_scatter")

    with fsdp_model.no_sync():
        root_module.calls.append("outer body")
        with fsdp_model.no_sync():
            root_module.calls.append("inner body")

    assert root_module.calls == [
        ("last_backward", False),
        ("all_reduce", False, True),
        "outer body",
        "inner body",
        ("all_reduce", True, True),
        ("last_backward", True),
    ]
    assert fsdp_model._no_sync_depth == 0


def test_fsdp2_partial_reduce_scatter_no_sync_requires_new_controls():
    """Fail before changing context depth when a required PyTorch control is absent."""
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)
    object.__setattr__(fsdp_model, "module", _SyncControlSpy())
    object.__setattr__(fsdp_model, "_no_sync_depth", 0)
    object.__setattr__(fsdp_model, "gradient_accumulation_mode", "partial_reduce_scatter")

    with pytest.raises(RuntimeError, match="set_requires_all_reduce"):
        with fsdp_model.no_sync():
            pass

    assert fsdp_model._no_sync_depth == 0


def test_fsdp2_validates_gradient_accumulation_mode(monkeypatch):
    """Reject unsupported or unsafe partial reduce-scatter configurations early."""
    monkeypatch.setattr(torch_fsdp2, "is_torch_min_version", lambda _version: True)
    _validate_fsdp_gradient_accumulation_mode("classic", False, False)
    _validate_fsdp_gradient_accumulation_mode("partial_reduce_scatter", True, True)

    with pytest.raises(ValueError, match="must be one of"):
        _validate_fsdp_gradient_accumulation_mode("unknown", True, True)
    with pytest.raises(ValueError, match="requires HSDP"):
        _validate_fsdp_gradient_accumulation_mode("partial_reduce_scatter", False, True)
    with pytest.raises(ValueError, match="reduce_scatter_unused_params=True"):
        _validate_fsdp_gradient_accumulation_mode("partial_reduce_scatter", True, False)

    monkeypatch.setattr(torch_fsdp2, "is_torch_min_version", lambda _version: False)
    with pytest.raises(RuntimeError, match="PyTorch >= 2.13"):
        _validate_fsdp_gradient_accumulation_mode("partial_reduce_scatter", True, True)


def test_fsdp2_scale_gradients_scales_shared_parameter_once():
    """Scale tied gradients once and leave missing gradients untouched."""

    class _SharedParameterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.nn.Parameter(torch.ones(4))
            self.weight = shared
            self.tied_weight = shared
            self.unused = torch.nn.Parameter(torch.ones(4))

    module = _SharedParameterModule()
    module.weight.grad = torch.full_like(module.weight, 4.0)
    fsdp_model = object.__new__(TorchFullyShardedDataParallel)
    object.__setattr__(fsdp_model, "module", module)

    fsdp_model.scale_gradients(torch.tensor(0.25))

    torch.testing.assert_close(module.weight.grad, torch.ones_like(module.weight))
    assert module.unused.grad is None


def test_fsdp2_wrap_plan_is_bottom_up_and_deduplicated():
    """Wrap nested modules once, excluding the root and preserving forward order."""
    model = _WrapRoot()

    forward_order, bottom_up_order = _build_fsdp_wrap_plan(model, _WRAP_TYPES)

    assert forward_order == [model.parent, model.parent.leaf]
    assert bottom_up_order == [model.parent.leaf, model.parent]
    assert model not in forward_order
    assert model not in bottom_up_order


@pytest.mark.parametrize(
    ("requested", "torch_supports_auto", "expected"),
    [
        (None, False, True),
        (None, True, None),
        (True, False, True),
        (False, False, False),
        (2, False, 2),
    ],
)
def test_fsdp2_resolves_auto_reshard_across_torch_versions(
    monkeypatch, requested, torch_supports_auto, expected
):
    """Map auto to the equivalent legacy policy without changing explicit values."""
    monkeypatch.setattr(torch_fsdp2, "is_torch_min_version", lambda _version: torch_supports_auto)

    resolved = torch_fsdp2._resolve_fsdp_reshard_after_forward(requested)

    assert resolved == expected
    assert type(resolved) is type(expected)


def test_fsdp2_output_view_clone_preserves_containers_and_repeated_references():
    """Clone dataclass view outputs once while preserving the no-op fast path."""
    base = torch.arange(8, dtype=torch.float32, requires_grad=True)
    view = base.view(2, 4)
    output = _ViewOutputContainer(view, (view,), "metadata")

    cloned = _clone_fsdp_output_views(torch.nn.Identity(), (), output)

    assert cloned is not output
    assert cloned.primary is cloned.aliases[0]
    assert cloned.primary._base is None
    assert cloned.metadata == output.metadata
    torch.testing.assert_close(cloned.primary, view)

    independent = (base.clone(),)
    assert _clone_fsdp_output_views(torch.nn.Identity(), (), independent) is independent


@pytest.mark.parametrize("reshard_after_forward", [None, True, False, 2])
def test_fsdp2_constructor_uses_bottom_up_wrap_plan(monkeypatch, reshard_after_forward):
    """Wrap children before parents while preserving forward-order prefetching."""
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    model = _WrapRoot()
    fake_mesh = object()
    fake_process_group = object()
    shard_calls = []
    prefetch_calls = []

    class _FakeDeviceMesh:
        @staticmethod
        def from_group(process_group, device_type):
            assert process_group is fake_process_group
            assert device_type == "cuda"
            return fake_mesh

    def fake_fully_shard(module, **kwargs):
        shard_calls.append((module, kwargs))
        module.set_modules_to_backward_prefetch = (
            lambda modules, module=module: prefetch_calls.append((module, modules))
        )

    monkeypatch.setattr(torch_fsdp2, "DeviceMesh", _FakeDeviceMesh)
    monkeypatch.setattr(torch_fsdp2, "fully_shard", fake_fully_shard)

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    config.recompute_granularity = "full"
    ddp_config = DistributedDataParallelConfig()
    ddp_config.reshard_after_forward = reshard_after_forward

    TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap=_WRAP_TYPES, process_group=fake_process_group
    )

    expected_reshard_after_forward = (
        True
        if reshard_after_forward is None and not is_torch_min_version("2.8.0")
        else reshard_after_forward
    )
    assert [module for module, _ in shard_calls] == [model.parent.leaf, model.parent, model]
    assert all(
        kwargs == {"mesh": fake_mesh, "reshard_after_forward": expected_reshard_after_forward}
        for _, kwargs in shard_calls
    )
    assert prefetch_calls == [(model.parent, []), (model.parent.leaf, [model.parent])]


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
@pytest.mark.parametrize("force_sum_supported", [False, True])
@pytest.mark.parametrize(
    "setter_name", ["set_gradient_divide_factor", "set_reduce_scatter_divide_factor"]
)
def test_fsdp2_constructor_configures_per_token_reduction(
    monkeypatch, calculate_per_token_loss, force_sum_supported, setter_name
):
    """Use true SUM when supported and compensate legacy AVG in per-token mode."""
    model = _WrapRoot()
    fake_mesh = object()

    class _FakeProcessGroup:
        @staticmethod
        def size():
            return 8

    fake_process_group = _FakeProcessGroup()
    shard_calls = []
    divide_factor_calls = []
    force_sum_calls = []

    class _FakeDeviceMesh:
        @staticmethod
        def from_group(process_group, device_type):
            assert process_group is fake_process_group
            assert device_type == "cuda"
            return fake_mesh

    def fake_fully_shard(module, **kwargs):
        shard_calls.append(module)
        setattr(
            module,
            setter_name,
            lambda factor, module=module: divide_factor_calls.append((module, factor)),
        )
        if force_sum_supported:
            setattr(
                module,
                "set_force_sum_reduction_for_comms",
                lambda force_sum, module=module: force_sum_calls.append((module, force_sum)),
            )

    monkeypatch.setattr(torch_fsdp2, "DeviceMesh", _FakeDeviceMesh)
    monkeypatch.setattr(torch_fsdp2, "fully_shard", fake_fully_shard)

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    config.calculate_per_token_loss = calculate_per_token_loss
    ddp_config = DistributedDataParallelConfig()

    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap=_WRAP_TYPES, process_group=fake_process_group
    )

    expected_modules = [model.parent.leaf, model.parent, model]
    assert shard_calls == expected_modules
    use_true_sum = calculate_per_token_loss and force_sum_supported
    expected_divide_calls = [(module, 1.0) for module in expected_modules] if use_true_sum else []
    expected_force_sum_calls = (
        [(module, True) for module in expected_modules] if use_true_sum else []
    )
    expected_correction = 8.0 if calculate_per_token_loss and not force_sum_supported else 1.0
    assert divide_factor_calls == expected_divide_calls
    assert force_sum_calls == expected_force_sum_calls
    assert fsdp_model._gradient_scale_correction == expected_correction


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("api_supported", [False, True])
def test_fsdp2_constructor_configures_unused_parameter_reduction(
    monkeypatch, enabled, api_supported
):
    """Opt in recursively and fail clearly when the PyTorch API is unavailable."""
    model = _WrapRoot()
    fake_mesh = object()
    fake_process_group = object()
    unused_param_calls = []

    class _FakeDeviceMesh:
        @staticmethod
        def from_group(process_group, device_type):
            assert process_group is fake_process_group
            assert device_type == "cuda"
            return fake_mesh

    def fake_fully_shard(module, **kwargs):
        if api_supported:
            setattr(
                module,
                "set_reduce_scatter_unused_params",
                lambda value, recurse, module=module: unused_param_calls.append(
                    (module, value, recurse)
                ),
            )

    monkeypatch.setattr(torch_fsdp2, "DeviceMesh", _FakeDeviceMesh)
    monkeypatch.setattr(torch_fsdp2, "fully_shard", fake_fully_shard)

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = TorchFullyShardedDataParallelConfig(reduce_scatter_unused_params=enabled)

    if enabled and not api_supported:
        with pytest.raises(RuntimeError, match="PyTorch >= 2.13"):
            TorchFullyShardedDataParallel(
                config,
                ddp_config,
                model,
                sub_modules_to_wrap=_WRAP_TYPES,
                process_group=fake_process_group,
            )
        return

    TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap=_WRAP_TYPES, process_group=fake_process_group
    )

    expected_calls = [(model, True, True)] if enabled else []
    assert unused_param_calls == expected_calls


def test_fsdp2_constructor_wraps_nested_modules(init_model_parallel):
    """Run forward and backward with real nested FSDP2 groups."""
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = DistributedDataParallelConfig()
    model = _NestedWrapRoot().cuda()

    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap={_WrapParent, _WrapLeaf, torch.nn.Linear}
    )
    output = fsdp_model(torch.randn(2, 8, device="cuda"))
    output.sum().backward()

    assert fsdp_model.module.__class__.__name__.startswith("FSDP")
    assert fsdp_model.module.parent.__class__.__name__.startswith("FSDP")
    assert fsdp_model.module.parent.leaf.__class__.__name__.startswith("FSDP")


def test_fsdp2_auto_reshard_keeps_only_root_unsharded_after_forward(init_model_parallel):
    """Apply PyTorch's automatic reshard policy to real nested FSDP2 groups."""
    if not is_torch_min_version("2.8.0"):
        pytest.skip("FSDP2 automatic reshard policy requires PyTorch >= 2.8.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = TorchFullyShardedDataParallelConfig(reshard_after_forward=None)
    model = _AutoReshardRoot().cuda()

    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap={_WrapLeaf}
    )
    output = fsdp_model(torch.randn(2, 8, device="cuda"))

    root_state = torch_fsdp2.fully_shard.state(fsdp_model.module)
    child_state = torch_fsdp2.fully_shard.state(fsdp_model.module.child)
    root_param_group = root_state._fsdp_param_group
    child_param_group = child_state._fsdp_param_group

    assert root_param_group is not None
    assert child_param_group is not None
    assert root_param_group.post_forward_mesh_info is None
    assert child_param_group.post_forward_mesh_info is not None
    assert root_param_group.is_unsharded
    assert not child_param_group.is_unsharded

    output.square().mean().backward()

    assert fsdp_model.module.root_bias.grad is not None
    assert fsdp_model.module.child.weight.grad is not None


def test_fsdp2_clones_output_views_before_registering_backward_hook(init_model_parallel):
    """Keep FSDP's pre-backward hook attached across downstream in-place ops."""
    if not is_torch_min_version("2.13.0"):
        pytest.skip("FSDP2 output-view safety warnings require PyTorch >= 2.13.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = TorchFullyShardedDataParallelConfig(clone_output_views=True)
    reference_model = _ViewOutputRoot().cuda()
    with torch.no_grad():
        torch.distributed.broadcast(reference_model.weight, src=0)
    model = _ViewOutputRoot().cuda()
    model.load_state_dict(reference_model.state_dict())
    fsdp_model = TorchFullyShardedDataParallel(config, ddp_config, model, sub_modules_to_wrap=set())
    group = fsdp_model.process_group
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    inputs = torch.arange(16, dtype=torch.float32, device="cuda").view(2, 8)
    inputs = inputs + rank / 10

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"FSDP2-wrapped module .* returned a view tensor")
        output = fsdp_model(inputs)
    assert output._base is None
    output.add_(0.25).square().mean().backward()

    reference_output = reference_model(inputs).clone()
    reference_output.add_(0.25).square().mean().backward()
    torch.distributed.all_reduce(reference_model.weight.grad, group=group)
    reference_model.weight.grad.div_(world_size)

    assert fsdp_model.module.weight.grad is not None
    torch.testing.assert_close(
        fsdp_model.module.weight.grad.full_tensor(),
        reference_model.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("gradient_accumulation_mode", ["classic", "partial_reduce_scatter"])
def test_fsdp2_hsdp_mesh_and_no_sync_match_global_reference(
    init_hsdp_model_parallel, gradient_accumulation_mode
):
    """Validate real 2x4 HSDP accumulation modes against full DP."""
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")
    if gradient_accumulation_mode == "partial_reduce_scatter" and not is_torch_min_version(
        "2.13.0"
    ):
        pytest.skip("Partial reduce-scatter accumulation requires PyTorch >= 2.13.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = TorchFullyShardedDataParallelConfig(
        num_distributed_optimizer_instances=2,
        gradient_accumulation_mode=gradient_accumulation_mode,
        reduce_scatter_unused_params=gradient_accumulation_mode == "partial_reduce_scatter",
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    reference_model = _NestedWrapRoot().cuda()
    with torch.no_grad():
        for param in reference_model.parameters():
            torch.distributed.broadcast(param, src=0, group=pg_collection.dp_cp)
    model = _NestedWrapRoot().cuda()
    model.load_state_dict(reference_model.state_dict())

    fsdp_model = TorchFullyShardedDataParallel(
        config,
        ddp_config,
        model,
        sub_modules_to_wrap={_WrapParent, _WrapLeaf, torch.nn.Linear},
        pg_collection=pg_collection,
    )
    assert fsdp_model.is_hsdp
    assert fsdp_model.process_group is pg_collection.dp_cp
    assert fsdp_model.shard_process_group is pg_collection.intra_dp_cp
    assert fsdp_model.replicate_process_group is pg_collection.inter_dist_opt
    assert tuple(fsdp_model.device_mesh.mesh.shape) == (2, 4)
    assert fsdp_model.device_mesh.mesh_dim_names == ("replicate", "shard")
    assert fsdp_model.device_mesh.get_group("replicate") is pg_collection.inter_dist_opt
    assert fsdp_model.device_mesh.get_group("shard") is pg_collection.intra_dp_cp
    fsdp_param = next(fsdp_model.module.parameters())
    assert isinstance(fsdp_param.placements[0], Replicate)
    assert isinstance(fsdp_param.placements[1], Shard)

    group = fsdp_model.process_group
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    for accumulation_cycle in range(2):
        fsdp_model.zero_grad(set_to_none=True)
        reference_model.zero_grad(set_to_none=True)
        microbatch_inputs = [
            torch.full(
                (2, 8), (rank + 1) * (microbatch + 1) * (accumulation_cycle + 1) / 10, device="cuda"
            )
            for microbatch in range(2)
        ]

        with fsdp_model.no_sync():
            fsdp_model(microbatch_inputs[0]).square().mean().backward()
        fsdp_modules = (fsdp_model.module.parent.leaf, fsdp_model.module.parent, fsdp_model.module)
        partial_reduce_outputs = _get_partial_reduce_outputs(*fsdp_modules)
        if gradient_accumulation_mode == "partial_reduce_scatter":
            assert partial_reduce_outputs
            assert all(output is not None for output in partial_reduce_outputs)
        else:
            assert all(output is None for output in partial_reduce_outputs)
        fsdp_model(microbatch_inputs[1]).square().mean().backward()
        assert all(output is None for output in _get_partial_reduce_outputs(*fsdp_modules))

        for microbatch_input in microbatch_inputs:
            reference_model(microbatch_input).square().mean().backward()
        for param in reference_model.parameters():
            torch.distributed.all_reduce(param.grad, group=group)
            param.grad.div_(world_size)

        fsdp_params = dict(fsdp_model.module.named_parameters())
        for name, reference_param in reference_model.named_parameters():
            fsdp_grad = fsdp_params[name].grad
            assert fsdp_grad is not None
            torch.testing.assert_close(
                fsdp_grad.full_tensor(), reference_param.grad, rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize(
    ("gradient_accumulation_mode", "calculate_per_token_loss"),
    [("classic", False), ("partial_reduce_scatter", True)],
)
def test_fsdp2_ep_hsdp_uses_distinct_meshes_and_matches_reference(
    init_ep_hsdp_model_parallel, gradient_accumulation_mode, calculate_per_token_loss
):
    """Reduce dense grads over DP8 and expert grads over expert-DP4 with a global divisor."""
    if not is_torch_min_version("2.13.0"):
        pytest.skip("Per-parameter FSDP2 meshes require PyTorch >= 2.13.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    config.expert_model_parallel_size = 2
    config.calculate_per_token_loss = calculate_per_token_loss
    ddp_config = TorchFullyShardedDataParallelConfig(
        num_distributed_optimizer_instances=2,
        gradient_accumulation_mode=gradient_accumulation_mode,
        reduce_scatter_unused_params=True,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    reference_model = _MixedDenseExpertRoot().cuda()
    ep_rank = pg_collection.ep.rank()
    with torch.no_grad():
        dense_values = torch.arange(64, device="cuda", dtype=torch.float32).view(8, 8)
        reference_model.dense_weight.copy_(dense_values / 100)
        reference_model.expert_weight.copy_(torch.eye(8, device="cuda") * (ep_rank + 1) / 10)
    model = _MixedDenseExpertRoot().cuda()
    model.load_state_dict(reference_model.state_dict())

    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap=set(), pg_collection=pg_collection
    )
    params = dict(fsdp_model.module.named_parameters())
    dense_param = params["dense_weight"]
    expert_param = params["expert_weight"]
    assert tuple(dense_param.device_mesh.mesh.shape) == (2, 4)
    assert tuple(expert_param.device_mesh.mesh.shape) == (2, 2)
    assert isinstance(dense_param.placements[0], Replicate)
    assert isinstance(dense_param.placements[1], Shard)
    assert isinstance(expert_param.placements[0], Replicate)
    assert isinstance(expert_param.placements[1], Shard)
    assert getattr(dense_param, "allreduce", True)
    assert not expert_param.allreduce

    fsdp_state = torch_fsdp2.fully_shard.state(fsdp_model.module)
    assert len(fsdp_state._fsdp_param_groups) == 2
    mesh_sizes = sorted(
        (group.mesh_info.replicate_process_group.size(), group.mesh_info.shard_process_group.size())
        for group in fsdp_state._fsdp_param_groups
    )
    assert mesh_sizes == [(2, 2), (2, 4)]

    dense_rank = pg_collection.dp_cp.rank()
    fsdp_model.zero_grad(set_to_none=True)
    reference_model.zero_grad(set_to_none=True)
    for microbatch in range(2):
        inputs = torch.full((2, 8), (dense_rank + 1) * (microbatch + 1) / 10, device="cuda")
        use_expert = (dense_rank // 2 + microbatch) % 2 == 0
        sync_context = fsdp_model.no_sync() if microbatch == 0 else nullcontext()
        with sync_context:
            fsdp_model(inputs, use_expert).square().mean().backward()
        reference_model(inputs, use_expert).square().mean().backward()

    torch.distributed.all_reduce(reference_model.dense_weight.grad, group=pg_collection.dp_cp)
    torch.distributed.all_reduce(reference_model.expert_weight.grad, group=pg_collection.expt_dp)
    if calculate_per_token_loss:
        scale = 1.0 / 16
        fsdp_model.scale_gradients(scale)
        reference_model.dense_weight.grad.mul_(scale)
        reference_model.expert_weight.grad.mul_(scale)
    else:
        reference_model.dense_weight.grad.div_(pg_collection.dp_cp.size())
        reference_model.expert_weight.grad.div_(pg_collection.dp_cp.size())

    torch.testing.assert_close(
        dense_param.grad.full_tensor(), reference_model.dense_weight.grad, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        expert_param.grad.full_tensor(), reference_model.expert_weight.grad, rtol=1e-5, atol=1e-6
    )


def test_fsdp2_ep_hsdp_params_norm_counts_each_global_parameter_once(init_ep_hsdp_model_parallel):
    """Combine dense and expert shard meshes without counting HSDP replicas."""
    if not is_torch_min_version("2.13.0"):
        pytest.skip("Per-parameter FSDP2 meshes require PyTorch >= 2.13.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    config.expert_model_parallel_size = 2
    ddp_config = TorchFullyShardedDataParallelConfig(
        num_distributed_optimizer_instances=2, reduce_scatter_unused_params=True
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model = _MixedDenseExpertRoot().cuda()
    with torch.no_grad():
        model.dense_weight.fill_(1.0)
        model.expert_weight.fill_(pg_collection.ep.rank() + 2.0)

    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap=set(), pg_collection=pg_collection
    )
    mock_args = SimpleNamespace(use_megatron_fsdp=False, bf16=True)
    with mock.patch("megatron.training.utils.common_utils.get_args", return_value=mock_args):
        actual_norm = training_utils.calc_params_l2_norm(fsdp_model, force_create_fp32_copy=True)

    # One dense weight of ones plus one global expert per EP rank, filled with 2 and 3.
    expected_norm = (64 * (1.0**2 + 2.0**2 + 3.0**2)) ** 0.5
    assert actual_norm == pytest.approx(expected_norm)


def test_fsdp2_no_sync_accumulates_and_reduces_gradients(init_model_parallel):
    """Match two-microbatch FSDP2 accumulation against an all-reduced reference."""
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = DistributedDataParallelConfig()
    reference_model = _NestedWrapRoot().cuda()
    with torch.no_grad():
        for param in reference_model.parameters():
            torch.distributed.broadcast(param, src=0)
    model = _NestedWrapRoot().cuda()
    model.load_state_dict(reference_model.state_dict())

    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, sub_modules_to_wrap={_WrapParent, _WrapLeaf, torch.nn.Linear}
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size(fsdp_model.process_group)

    for accumulation_cycle in range(2):
        fsdp_model.zero_grad(set_to_none=True)
        reference_model.zero_grad(set_to_none=True)
        microbatch_inputs = [
            torch.full(
                (2, 8), (rank + 1) * (microbatch + 1) * (accumulation_cycle + 1) / 10, device="cuda"
            )
            for microbatch in range(2)
        ]

        with fsdp_model.no_sync():
            fsdp_model(microbatch_inputs[0]).square().mean().backward()
        fsdp_model(microbatch_inputs[1]).square().mean().backward()

        for microbatch_input in microbatch_inputs:
            reference_model(microbatch_input).square().mean().backward()
        for param in reference_model.parameters():
            torch.distributed.all_reduce(param.grad, group=fsdp_model.process_group)
            param.grad.div_(world_size)

        fsdp_params = dict(fsdp_model.module.named_parameters())
        for name, reference_param in reference_model.named_parameters():
            fsdp_grad = fsdp_params[name].grad
            assert fsdp_grad is not None
            torch.testing.assert_close(
                fsdp_grad.full_tensor(), reference_param.grad, rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize(
    ("parallel_fixture", "gradient_accumulation_mode"),
    [
        pytest.param("init_model_parallel", "classic", id="fsdp1d-classic"),
        pytest.param("init_hsdp_model_parallel", "partial_reduce_scatter", id="hsdp-partial"),
    ],
)
def test_fsdp2_rank_divergent_unused_parameters_match_global_reference(
    request, parallel_fixture, gradient_accumulation_mode
):
    """Match rank-divergent unused gradients for 1D classic and HSDP partial modes."""
    request.getfixturevalue(parallel_fixture)
    use_partial_reduce_scatter = gradient_accumulation_mode == "partial_reduce_scatter"

    if not is_torch_min_version("2.13.0"):
        pytest.skip("Unused-parameter-safe FSDP2 reduce-scatter requires PyTorch >= 2.13.")
    if torch.distributed.get_world_size() != 8:
        pytest.skip("This numerical test requires exactly 8 ranks.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = TorchFullyShardedDataParallelConfig(
        reduce_scatter_unused_params=True,
        num_distributed_optimizer_instances=2 if use_partial_reduce_scatter else 1,
        gradient_accumulation_mode=gradient_accumulation_mode,
    )
    pg_collection = (
        ProcessGroupCollection.use_mpu_process_groups() if use_partial_reduce_scatter else None
    )
    reference_model = _ConditionalParamRoot().cuda()
    with torch.no_grad():
        for param in reference_model.parameters():
            torch.distributed.broadcast(param, src=0)
    model = _ConditionalParamRoot().cuda()
    model.load_state_dict(reference_model.state_dict())

    fsdp_model = TorchFullyShardedDataParallel(
        config,
        ddp_config,
        model,
        sub_modules_to_wrap={_ConditionalParamBlock},
        pg_collection=pg_collection,
    )
    assert fsdp_model.is_hsdp is use_partial_reduce_scatter
    group = fsdp_model.process_group
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    microbatch_inputs = [
        torch.full((2, 8), (rank + 1) * (microbatch + 1) / 10, device="cuda")
        for microbatch in range(2)
    ]
    branch_flags = [(rank < 4, False, False), (False, 2 <= rank < 6, rank % 2 == 0)]

    fsdp_model.zero_grad(set_to_none=True)
    reference_model.zero_grad(set_to_none=True)
    with fsdp_model.no_sync():
        fsdp_model(microbatch_inputs[0], *branch_flags[0]).square().mean().backward()
    if use_partial_reduce_scatter:
        partial_reduce_outputs = _get_partial_reduce_outputs(
            fsdp_model.module.block, fsdp_model.module
        )
        assert partial_reduce_outputs
        assert all(output is not None for output in partial_reduce_outputs)
    fsdp_model(microbatch_inputs[1], *branch_flags[1]).square().mean().backward()
    if use_partial_reduce_scatter:
        assert all(
            output is None
            for output in _get_partial_reduce_outputs(fsdp_model.module.block, fsdp_model.module)
        )

    for microbatch_input, flags in zip(microbatch_inputs, branch_flags):
        reference_model(microbatch_input, *flags).square().mean().backward()

    if rank >= 4:
        assert reference_model.block.branch_a.weight.grad is None
    if not 2 <= rank < 6:
        assert reference_model.block.branch_b.weight.grad is None
    if rank % 2 == 0:
        assert reference_model.root_branch.grad is not None
    else:
        assert reference_model.root_branch.grad is None
    assert reference_model.block.explicit_zero.grad is not None
    assert torch.count_nonzero(reference_model.block.explicit_zero.grad) == 0
    assert reference_model.block.globally_unused.grad is None
    assert reference_model.block.frozen.grad is None
    assert reference_model.root_globally_unused.grad is None

    expected_grads = {}
    for name, param in reference_model.named_parameters():
        if not param.requires_grad:
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.clone()
        torch.distributed.all_reduce(grad, group=group)
        grad.div_(world_size)
        expected_grads[name] = grad
    assert torch.count_nonzero(expected_grads["block.branch_a.weight"]) > 0

    fsdp_params = dict(fsdp_model.module.named_parameters())
    actual_grads = {}
    for name, param in fsdp_params.items():
        if not param.requires_grad:
            assert param.grad is None
            continue
        assert param.grad is not None
        actual_grad = param.grad.full_tensor()
        actual_grads[name] = actual_grad
        torch.testing.assert_close(actual_grad, expected_grads[name], rtol=1e-5, atol=1e-6)

    assert torch.count_nonzero(actual_grads["block.explicit_zero"]) == 0
    assert torch.count_nonzero(actual_grads["block.globally_unused"]) == 0
    assert torch.count_nonzero(actual_grads["root_globally_unused"]) == 0


@pytest.mark.parametrize(
    ("parallel_fixture", "gradient_accumulation_mode"),
    [
        pytest.param("init_model_parallel", "classic", id="fsdp1d-classic"),
        pytest.param("init_hsdp_model_parallel", "partial_reduce_scatter", id="hsdp-partial"),
    ],
)
@pytest.mark.parametrize("all_zero_tokens", [False, True])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
def test_fsdp2_per_token_gradient_scaling_matches_global_reference(
    request, parallel_fixture, gradient_accumulation_mode, all_zero_tokens, model_dtype
):
    """Match per-token SUM normalization for 1D classic and HSDP partial modes."""
    request.getfixturevalue(parallel_fixture)
    use_partial_reduce_scatter = gradient_accumulation_mode == "partial_reduce_scatter"
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")
    if use_partial_reduce_scatter and not is_torch_min_version("2.13.0"):
        pytest.skip("Partial reduce-scatter accumulation requires PyTorch >= 2.13.")
    if torch.distributed.get_world_size() != 8:
        pytest.skip("This numerical test requires exactly 8 ranks.")

    config = TransformerConfig(
        num_layers=1, kv_channels=1, bf16=True, calculate_per_token_loss=True
    )
    ddp_config = TorchFullyShardedDataParallelConfig(
        average_in_collective=False,
        num_distributed_optimizer_instances=2 if use_partial_reduce_scatter else 1,
        gradient_accumulation_mode=gradient_accumulation_mode,
        reduce_scatter_unused_params=use_partial_reduce_scatter,
    )
    pg_collection = (
        ProcessGroupCollection.use_mpu_process_groups() if use_partial_reduce_scatter else None
    )
    group = (
        pg_collection.dp_cp
        if pg_collection is not None
        else parallel_state.get_data_parallel_group(with_context_parallel=True)
    )
    reference_model = _NestedWrapRoot().cuda().to(model_dtype)
    with torch.no_grad():
        for param in reference_model.parameters():
            torch.distributed.broadcast(param, src=0, group=group)
    model = _NestedWrapRoot().cuda().to(model_dtype)
    model.load_state_dict(reference_model.state_dict())

    fsdp_model = TorchFullyShardedDataParallel(
        config,
        ddp_config,
        model,
        sub_modules_to_wrap={_WrapParent, _WrapLeaf, torch.nn.Linear},
        pg_collection=pg_collection,
    )
    assert fsdp_model.is_hsdp is use_partial_reduce_scatter
    assert fsdp_model.process_group is group
    rank = torch.distributed.get_rank(group)
    token_counts = [0, 0] if all_zero_tokens else [rank % 4, (3 * rank + 1) % 5]
    token_ids = torch.arange(4, device="cuda")
    base_inputs = torch.arange(32, dtype=torch.float32, device="cuda").view(4, 8)
    microbatch_inputs = [
        (base_inputs + 1 + rank * 0.125 + microbatch * 0.25) / 50 for microbatch in range(2)
    ]
    microbatch_inputs = [inputs.to(model_dtype) for inputs in microbatch_inputs]

    def token_sum_loss(module, inputs, num_valid_tokens):
        output = module(inputs)
        valid_mask = (token_ids < num_valid_tokens).to(output.dtype)
        return (output.square().sum(dim=-1) * valid_mask).sum()

    fsdp_model.zero_grad(set_to_none=True)
    reference_model.zero_grad(set_to_none=True)
    with fsdp_model.no_sync():
        token_sum_loss(fsdp_model, microbatch_inputs[0], token_counts[0]).backward()
    if use_partial_reduce_scatter:
        partial_reduce_outputs = _get_partial_reduce_outputs(
            fsdp_model.module.parent.leaf, fsdp_model.module.parent, fsdp_model.module
        )
        assert partial_reduce_outputs
        assert all(output is not None for output in partial_reduce_outputs)
    token_sum_loss(fsdp_model, microbatch_inputs[1], token_counts[1]).backward()
    if use_partial_reduce_scatter:
        assert all(
            output is None
            for output in _get_partial_reduce_outputs(
                fsdp_model.module.parent.leaf, fsdp_model.module.parent, fsdp_model.module
            )
        )

    for microbatch_input, token_count in zip(microbatch_inputs, token_counts):
        token_sum_loss(reference_model, microbatch_input, token_count).backward()

    local_num_tokens = torch.tensor(sum(token_counts), dtype=torch.int, device="cuda")
    expected_num_tokens = local_num_tokens.clone()
    torch.distributed.all_reduce(expected_num_tokens, group=group)

    finalized_num_tokens = local_num_tokens.clone()
    finalize_model_grads([fsdp_model], num_tokens=finalized_num_tokens)

    assert torch.equal(finalized_num_tokens, expected_num_tokens)
    assert expected_num_tokens.item() == (0 if all_zero_tokens else 29)

    safe_num_tokens = torch.clamp(expected_num_tokens, min=1)
    for param in reference_model.parameters():
        assert param.grad is not None
        torch.distributed.all_reduce(param.grad, group=group)
        param.grad.div_(safe_num_tokens)

    fsdp_params = dict(fsdp_model.module.named_parameters())
    for name, reference_param in reference_model.named_parameters():
        fsdp_grad = fsdp_params[name].grad
        assert fsdp_grad is not None
        actual_grad = fsdp_grad.full_tensor()
        assert torch.isfinite(actual_grad).all()
        # FSDP reduce-scatter and the reference all-reduce use different NCCL
        # reduction trees, so allow a few BF16 ULPs after microbatch accumulation.
        rtol = 2e-2 if model_dtype == torch.bfloat16 else 1e-5
        atol = 1e-2 if model_dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(actual_grad, reference_param.grad, rtol=rtol, atol=atol)
        if all_zero_tokens:
            assert torch.count_nonzero(actual_grad) == 0


def test_fsdp2_constructor(init_model_parallel):
    """Test the FSDP2 constructor."""
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    # Create a dummy model and configs.
    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = DistributedDataParallelConfig()
    model = DummyModel(config)
    model = Float16Module(config, model)
    ddp_config = DistributedDataParallelConfig()

    # Create the sharded model.
    fsdp_model = TorchFullyShardedDataParallel(config, ddp_config, model)

    def _is_fsdp_wrapped_module(instance):
        # FSDP adds a prefix to the class name.
        return instance.__class__.__name__.startswith("FSDP")

    assert isinstance(fsdp_model, TorchFullyShardedDataParallel)
    # We manually added Linear to the list of submodules to wrap.
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.linear)
    # ColumnParallelLinear is in the default list of submodules to wrap.
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.column_parallel_linear)
    # Conv2d is not in the list of submodules to wrap.
    assert not _is_fsdp_wrapped_module(fsdp_model.module.module.conv)


def test_fsdp2_constructor_with_process_group(init_model_parallel):
    """Test the FSDP2 constructor with explicit process group parameter."""
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    # Create a dummy model and configs.
    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = DistributedDataParallelConfig()
    model = DummyModel(config)
    model = Float16Module(config, model)

    # Create a custom process group (using the default world for testing)
    custom_process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    # Create the sharded model with explicit process group
    fsdp_model = TorchFullyShardedDataParallel(
        config, ddp_config, model, process_group=custom_process_group
    )

    # Verify the process group was set correctly
    assert fsdp_model.process_group is custom_process_group

    # Check that module wrapping still works correctly
    def _is_fsdp_wrapped_module(instance):
        return instance.__class__.__name__.startswith("FSDP")

    assert isinstance(fsdp_model, TorchFullyShardedDataParallel)
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.linear)
    assert _is_fsdp_wrapped_module(fsdp_model.module.module.column_parallel_linear)
    assert not _is_fsdp_wrapped_module(fsdp_model.module.module.conv)
