# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

import megatron.core.distributed.torch_fully_sharded_data_parallel as torch_fsdp2
from megatron.core import parallel_state
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.torch_fully_sharded_data_parallel import (
    TorchFullyShardedDataParallel,
    _build_fsdp_wrap_plan,
)
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
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


class _WrapRoot(torch.nn.Module):
    """Root matching a wrap type and sharing a nested child by identity."""

    def __init__(self):
        super().__init__()
        self.parent = _WrapParent()
        self.shared_leaf = self.parent.leaf


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


def test_fsdp2_wrap_plan_is_bottom_up_and_deduplicated():
    """Wrap nested modules once, excluding the root and preserving forward order."""
    model = _WrapRoot()

    forward_order, bottom_up_order = _build_fsdp_wrap_plan(model, _WRAP_TYPES)

    assert forward_order == [model.parent, model.parent.leaf]
    assert bottom_up_order == [model.parent.leaf, model.parent]
    assert model not in forward_order
    assert model not in bottom_up_order


@pytest.mark.parametrize("reshard_after_forward", [True, False, 2])
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

    assert [module for module, _ in shard_calls] == [model.parent.leaf, model.parent, model]
    assert all(
        kwargs == {"mesh": fake_mesh, "reshard_after_forward": reshard_after_forward}
        for _, kwargs in shard_calls
    )
    assert prefetch_calls == [(model.parent, []), (model.parent.leaf, [model.parent])]


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


def test_fsdp2_no_sync_accumulates_and_reduces_gradients(init_model_parallel):
    """Match two-microbatch FSDP2 accumulation against an all-reduced reference."""
    if not is_torch_min_version("2.6.0"):
        pytest.skip("FSDP2 is not supported on this version of PyTorch.")

    config = TransformerConfig(num_layers=1, kv_channels=1, bf16=True)
    ddp_config = DistributedDataParallelConfig()
    reference_model = _NestedWrapRoot().cuda()
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
