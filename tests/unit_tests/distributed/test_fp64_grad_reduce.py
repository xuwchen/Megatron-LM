# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU numerical references for gradient communication and norm accumulation."""

import argparse
import itertools
import math

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fp64_grad_reduce import (
    GradientReductionWorkGroup,
    all_reduce_fp64,
    reduce_scatter_fp64,
)
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32
from megatron.training.arguments import add_megatron_arguments
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def distributed_context():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def cancellation_inputs():
    """Every summation order encounters both large cancellation and small terms."""
    if dist.get_world_size() % 4:
        pytest.skip("Requires a world size divisible by four")
    coefficients = [2**24, 1, -(2**24), 1]
    permutations = list(itertools.permutations(range(4)))
    inputs = [
        torch.tensor([coefficients[p[rank % 4]] for p in permutations], dtype=torch.float32).repeat(
            dist.get_world_size() * 3, 1
        )
        for rank in range(dist.get_world_size())
    ]
    reference = sum(value.double() for value in inputs)
    return inputs, reference


@pytest.mark.parametrize("operation", ["all_reduce", "reduce_scatter"])
@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_collective_against_independent_fp64_sum(operation, mean, asynchronous):
    inputs, reference = cancellation_inputs()
    if mean:
        reference /= dist.get_world_size()
    op = dist.ReduceOp.AVG if mean else dist.ReduceOp.SUM
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        grad = inputs[dist.get_rank()].cuda()
        if operation == "all_reduce":
            result = grad
            handle = all_reduce_fp64(grad, op=op, group=dist.group.WORLD, async_op=asynchronous)
        else:
            result = torch.empty_like(grad.chunk(dist.get_world_size())[0])
            handle = reduce_scatter_fp64(
                result, grad, op=op, group=dist.group.WORLD, async_op=asynchronous
            )
            reference = reference.chunk(dist.get_world_size())[dist.get_rank()]
    torch.cuda.current_stream().wait_stream(producer)
    if asynchronous:
        handle.wait()
        handle.wait()  # Mean division and roundback must be idempotent.
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), reference.float(), atol=0, rtol=0)


def test_multiple_deferred_copies():
    inputs, reference = cancellation_inputs()
    a = inputs[dist.get_rank()].cuda()
    b = -a.clone()
    group = GradientReductionWorkGroup(
        [
            all_reduce_fp64(a, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=True),
            all_reduce_fp64(b, op=dist.ReduceOp.AVG, group=dist.group.WORLD, async_op=True),
        ]
    )
    group.wait()
    group.wait()
    torch.testing.assert_close(a.cpu(), reference.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        b.cpu(), (-reference / dist.get_world_size()).float(), atol=0, rtol=0
    )


def test_norm_and_clipping_across_gradient_partitions():
    full = (torch.arange(300_003, dtype=torch.float32) % 101) * 0.001
    full[::100003] = 1000.0
    expected = math.sqrt(math.fsum(value * value for value in full.tolist()))
    owned_whole = [full.cuda()] if dist.get_rank() == 0 else []
    local = full.chunk(dist.get_world_size())[dist.get_rank()].cuda()
    whole_norm = get_grad_norm_fp32(
        owned_whole, grad_stats_parallel_group=dist.group.WORLD, use_fp64=True
    )
    sharded_norm = get_grad_norm_fp32(
        [local], grad_stats_parallel_group=dist.group.WORLD, use_fp64=True
    )
    assert whole_norm == pytest.approx(expected, rel=5e-14)
    assert sharded_norm == pytest.approx(expected, rel=5e-14)
    parameter = torch.nn.Parameter(torch.zeros_like(local))
    parameter.grad = local.clone()
    clip_grad_by_total_norm_fp32([parameter], max_norm=1.0, total_norm=sharded_norm)
    coefficient = torch.tensor(1.0 / (expected + 1e-6), dtype=torch.float32, device="cuda")
    torch.testing.assert_close(parameter.grad, local * coefficient, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grad_reduce_in_fp32": False},
        {"grad_reduce_in_fp32": True, "num_distributed_optimizer_instances": 2},
        {"grad_reduce_in_fp32": True, "reduce_scatter_with_fp32_accumulation": True},
    ],
)
def test_configuration_rejects_unsupported_reduction_paths(kwargs):
    with pytest.raises(ValueError, match="grad_reduce_in_fp64"):
        DistributedDataParallelConfig(grad_reduce_in_fp64=True, **kwargs)


def test_collective_rejects_wrong_storage_dtype():
    with pytest.raises(ValueError, match="FP32"):
        all_reduce_fp64(
            torch.ones(4, device="cuda", dtype=torch.bfloat16),
            op=dist.ReduceOp.SUM,
            group=dist.group.WORLD,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_training_cli_precision_options(enabled):
    parser = add_megatron_arguments(argparse.ArgumentParser())
    argv = ["--grad-reduce-in-fp64", "--grad-norm-in-fp64"] if enabled else []
    args = parser.parse_args(argv)
    assert args.grad_reduce_in_fp64 is enabled
    assert args.grad_norm_in_fp64 is enabled
