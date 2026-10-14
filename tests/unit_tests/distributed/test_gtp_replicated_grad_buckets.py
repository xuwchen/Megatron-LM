# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import weakref

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from megatron.core.distributed.finalize_model_grads import (
    _allreduce_replicated_grads_over_gtp_remat_group,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def groups():
    Utils.initialize_distributed()
    world = dist.get_world_size()
    assert world >= 4 and world % 2 == 0
    split = world // 2
    expert_groups = [dist.new_group(list(range(start, start + split))) for start in (0, split)]
    yield dist.group.WORLD, expert_groups[dist.get_rank() // split]
    for group in expert_groups:
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)


@pytest.mark.parametrize('per_token_loss', [False, True])
@pytest.mark.parametrize('bucket_bytes', [None, 32, 1])
@pytest.mark.parametrize('mixed_dtype', [False, True])
def test_replicated_grad_buckets_match_individual_reductions(
    groups, monkeypatch, per_token_loss, bucket_bytes, mixed_dtype
):
    dense_group, expert_group = groups
    model = torch.nn.Module()
    expected, pointers, skipped = {}, {}, {}
    op = dist.ReduceOp.SUM if per_token_loss else dist.ReduceOp.AVG
    for index, shape in enumerate(((2, 3), (5, 8), (0,), (3, 2), (4, 4), (2, 3))):
        param = torch.nn.Parameter(torch.zeros(shape, device='cuda'))
        param.allreduce = index < 3
        param.grad = torch.full_like(param, 123)
        dtype = torch.float16 if mixed_dtype and index % 2 else torch.float32
        grad = torch.arange(param.numel(), device='cuda', dtype=dtype).reshape(shape)
        grad = grad + dist.get_rank() + 1
        if index == 0:
            grad = grad.t().contiguous().t()  # Exercise noncontiguous main_grad.
            assert not grad.is_contiguous()
        param.main_grad = grad
        model.register_parameter(f'p{index}', param)
        reference = grad.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(reference, op=op, group=dense_group if param.allreduce else expert_group)
        expected[f'p{index}'] = reference
        pointers[f'p{index}'] = grad.data_ptr()

    for name in ('sharded', 'frozen', 'missing'):
        param = torch.nn.Parameter(torch.zeros(3, device='cuda'), requires_grad=name != 'frozen')
        param.grad = torch.full_like(param, 123)
        param.main_grad = None if name == 'missing' else torch.full_like(param, 17)
        param.is_gtp_weight_remat = name == 'sharded'
        model.register_parameter(name, param)
        skipped[name] = None if param.main_grad is None else param.main_grad.clone()

    calls, previous_tensors = [], []
    original_all_reduce = dist.all_reduce

    def record_all_reduce(tensor, *, op, group):
        # A retained coalesced tensor would overlap scratch with the next allocation.
        assert all(ref() is None for ref in previous_tensors)
        previous_tensors.append(weakref.ref(tensor))
        calls.append((tensor.numel() * tensor.element_size(), group))
        return original_all_reduce(tensor, op=op, group=group)

    monkeypatch.setattr(dist, 'all_reduce', record_all_reduce)
    _allreduce_replicated_grads_over_gtp_remat_group(
        [model], dense_group, expert_group, per_token_loss, bucket_size_bytes=bucket_bytes
    )
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, torch.full_like(param, 123), rtol=0, atol=0)
        if name in expected:
            torch.testing.assert_close(param.main_grad, expected[name], rtol=0, atol=0)
            assert param.main_grad.data_ptr() == pointers[name]
        elif skipped[name] is None:
            assert param.main_grad is None
        else:
            torch.testing.assert_close(param.main_grad, skipped[name], rtol=0, atol=0)
    assert {group for _, group in calls} == {dense_group, expert_group}
    if bucket_bytes is None:
        assert len(calls) == 2
    else:
        largest = max(t.numel() * t.element_size() for t in expected.values())
        assert max(size for size, _ in calls) <= max(bucket_bytes, largest)
        assert len(calls) > 2


@pytest.mark.parametrize('per_token_loss', [False, True])
def test_bucketed_reduction_preserves_dtensor_placement(groups, per_token_loss):
    dense_group, _ = groups
    mesh = DeviceMesh('cuda', list(range(dist.get_world_size())))
    local = torch.full((2, 3), float(dist.get_rank() + 1), device='cuda')
    grad = DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
    model = torch.nn.Module()
    param = torch.nn.Parameter(torch.zeros(grad.shape, device='cuda'))
    param.main_grad = grad
    model.register_parameter('weight', param)
    _allreduce_replicated_grads_over_gtp_remat_group(
        [model], dense_group, None, per_token_loss, bucket_size_bytes=16
    )
    assert isinstance(param.main_grad, DTensor)
    assert param.main_grad.placements == grad.placements
    assert param.main_grad.device_mesh == mesh
    factor = dist.get_world_size() if per_token_loss else 1
    torch.testing.assert_close(param.main_grad.to_local(), local * factor, rtol=0, atol=0)


@pytest.mark.parametrize('bucket_mb', [0, -1])
def test_invalid_replicated_grad_bucket_size(bucket_mb):
    with pytest.raises(ValueError, match='gtp_replicated_grad_bucket_size_mb must be positive'):
        ModelParallelConfig(gtp_replicated_grad_bucket_size_mb=bucket_mb)
