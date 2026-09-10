# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP32 norm references and exact invariance under arbitrary optimizer shards."""

import argparse

import pytest
import torch
import torch.distributed as dist

from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.optimizer.fixed_grad_norm import (
    METADATA_ATTR,
    FixedGradNormMetadata,
    get_fixed_grad_norm_fp32,
)
from megatron.core.optimizer.optimizer import copy_optimizer_param_metadata
from megatron.training.arguments import add_megatron_arguments
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def distributed_context():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _values(numel):
    index = torch.arange(numel, dtype=torch.float32)
    return (index.remainder(19) - 9) * torch.pow(2.0, index.remainder(23) - 11)


def _shard(full, layout):
    rank, size = dist.get_rank(), dist.get_world_size()
    result = []
    for index, (name, values) in enumerate(full.items()):
        numel = values.numel()
        if layout == 'whole':
            if rank != index % size:
                continue
            start, end = 0, numel
        else:
            # Deliberately split inside logical blocks, and rotate ownership.
            owner = rank if layout == 'split' else (rank + 1) % size
            boundaries = [0] + [max(0, numel * i // size - 7) for i in range(1, size)] + [numel]
            start, end = boundaries[owner : owner + 2]
        grad = values[start:end].cuda()
        if end == numel and end > start:
            # Nonzero storage padding must not contribute to the logical norm.
            grad = torch.cat([grad, torch.full((13,), 1e10, device='cuda', dtype=torch.float32)])
        setattr(grad, METADATA_ATTR, FixedGradNormMetadata(name, start, numel))
        result.append(grad)
    return result if layout == 'whole' else list(reversed(result))


@pytest.mark.parametrize('numel', [1, 31, 1023, 1024, 1025, 8193])
def test_norm_is_invariant_to_shards_and_parameter_order(numel):
    full = {'z_matrix': _values(numel), 'a_vector': _values(107), 'm_matrix': _values(4096)}
    # An independent native FP32 L2 calculation, without the implementation's
    # logical-block decomposition or any FP64 norm operation.
    reference = torch.linalg.vector_norm(torch.cat(list(full.values())))
    results = [
        get_fixed_grad_norm_fp32(_shard(full, layout), dist.group.WORLD)
        for layout in ['whole', 'split', 'rotated']
    ]
    for result in results:
        assert result.dtype == torch.float32
        torch.testing.assert_close(result.cpu().squeeze(), reference, rtol=2e-6, atol=1e-6)
        torch.testing.assert_close(result, results[0], rtol=0, atol=0)
    integrated = get_grad_norm_fp32(
        _shard(full, 'rotated'), grad_stats_parallel_group=dist.group.WORLD, use_fixed_order=True
    )
    torch.testing.assert_close(integrated, results[0], rtol=0, atol=0)


def test_all_empty_gradients_have_zero_norm():
    norm = get_fixed_grad_norm_fp32([], dist.group.WORLD)
    assert norm.dtype == torch.float32 and norm.item() == 0.0


def test_optimizer_metadata_preserves_shard_offsets():
    param = torch.nn.Parameter(torch.ones(50, device='cuda'))
    setattr(param, METADATA_ATTR, FixedGradNormMetadata('example', 100, 300))
    main = param.detach()[7:25].clone()
    copy_optimizer_param_metadata(main, param, shard_start=7)
    assert getattr(main, METADATA_ATTR) == FixedGradNormMetadata('example', 107, 300)
    grad = torch.zeros_like(main)
    copy_optimizer_param_metadata(grad, main)
    assert getattr(grad, METADATA_ATTR) == getattr(main, METADATA_ATTR)


@pytest.mark.parametrize('problem', ['missing', 'overlap', 'size'])
def test_invalid_global_ownership_is_rejected(problem):
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size > 1
    numel = 100 * size
    start, length, total = rank * 100, 100, numel
    if problem == 'missing' and rank == 0:
        start, length = 1, 99
    elif problem == 'overlap' and rank == 1:
        start = 99
    elif problem == 'size':
        total += rank
    grad = torch.ones(length, device='cuda', dtype=torch.float32)
    setattr(grad, METADATA_ATTR, FixedGradNormMetadata('invalid', start, total))
    with pytest.raises(ValueError, match='ownership|logical gradient size'):
        get_fixed_grad_norm_fp32([grad], dist.group.WORLD)


@pytest.mark.parametrize('problem', ['metadata', 'dtype', 'fp64', 'norm_type'])
def test_unsupported_inputs_are_rejected(problem):
    grad = torch.ones(
        10, device='cuda', dtype=torch.bfloat16 if problem == 'dtype' else torch.float32
    )
    if problem != 'metadata':
        setattr(grad, METADATA_ATTR, FixedGradNormMetadata('test', 0, 10))
    with pytest.raises(ValueError):
        get_grad_norm_fp32(
            [grad],
            grad_stats_parallel_group=dist.group.WORLD,
            use_fixed_order=True,
            use_fp64=problem == 'fp64',
            norm_type=1 if problem == 'norm_type' else 2,
        )


@pytest.mark.parametrize('enabled', [False, True])
def test_training_parser_keeps_both_fp64_options_off(enabled):
    parser = add_megatron_arguments(argparse.ArgumentParser())
    args = parser.parse_args(['--grad-norm-in-fixed-order'] if enabled else [])
    assert args.grad_norm_in_fixed_order is enabled
    assert args.grad_norm_in_fp64 is False
    assert args.grad_reduce_in_fp64 is False
