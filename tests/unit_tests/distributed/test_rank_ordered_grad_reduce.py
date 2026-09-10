# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent FP32 references and layout invariance for ordered collectives."""

import argparse
import itertools
import struct

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.rank_ordered_grad_reduce import (
    RankOrderedReductionWorkGroup,
    all_reduce_rank_ordered,
    reduce_scatter_rank_ordered,
)
from megatron.training.arguments import add_megatron_arguments
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def distributed_context():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def rounded(value):
    """Round a scalar through its IEEE binary32 encoding."""
    return struct.unpack("f", struct.pack("f", value))[0]


def inputs_and_reference():
    """Use cancellation cases with an independently rounded scalar left fold."""
    if dist.get_world_size() != 4:
        pytest.skip("The independent reference exercises four-rank layouts")
    permutations = list(itertools.permutations([2**24, 1, -(2**24), 1]))
    inputs = torch.tensor(permutations, dtype=torch.float32).T.repeat(1, 8)
    reference = []
    for values in permutations * 8:
        total = values[0]
        for value in values[1:]:
            total = rounded(total + value)
        reference.append(total)
    return inputs, torch.tensor(reference, dtype=torch.float32)


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_independent_fp32_reference(scatter, mean, asynchronous):
    inputs, reference = inputs_and_reference()
    producer = torch.cuda.Stream()
    op = dist.ReduceOp.AVG if mean else dist.ReduceOp.SUM
    if mean:
        reference /= dist.get_world_size()
    with torch.cuda.stream(producer):
        source = inputs[dist.get_rank()].cuda()
        if scatter:
            output = torch.empty_like(source.chunk(dist.get_world_size())[0])
            work = reduce_scatter_rank_ordered(
                output, source, op=op, group=dist.group.WORLD, async_op=asynchronous
            )
            reference = reference.chunk(dist.get_world_size())[dist.get_rank()]
        else:
            output = source
            work = all_reduce_rank_ordered(
                source, op=op, group=dist.group.WORLD, async_op=asynchronous
            )
    torch.cuda.current_stream().wait_stream(producer)
    if asynchronous:
        assert work.source.dtype == work.received.dtype == torch.float32
        work.wait()
        work.wait()
    assert output.dtype == torch.float32
    torch.testing.assert_close(output.cpu(), reference, atol=0, rtol=0)


def test_bucket_offsets_and_shard_ownership_do_not_change_results():
    inputs, reference = inputs_and_reference()
    raw = inputs[dist.get_rank()].cuda()
    whole = torch.cat([torch.ones(37, device="cuda"), raw, torch.ones(6, device="cuda")])
    # 235 elements requires padding before the four-way exchange.
    all_reduce_rank_ordered(whole, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    shard = torch.empty_like(raw.chunk(dist.get_world_size())[0])
    reduce_scatter_rank_ordered(shard, raw, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
    assert torch.equal(whole[37:-6].cpu(), reference)
    assert torch.equal(shard.cpu(), reference.chunk(dist.get_world_size())[dist.get_rank()])
    assert torch.equal(whole[37:-6].chunk(dist.get_world_size())[dist.get_rank()], shard)


def test_deferred_all_reduces_preserve_collective_issue_order():
    inputs, reference = inputs_and_reference()
    a = inputs[dist.get_rank()].cuda()
    b = a.clone()
    group = RankOrderedReductionWorkGroup(
        [
            all_reduce_rank_ordered(a, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=True),
            all_reduce_rank_ordered(b, op=dist.ReduceOp.AVG, group=dist.group.WORLD, async_op=True),
        ]
    )
    group.wait()
    group.wait()
    torch.testing.assert_close(a.cpu(), reference, atol=0, rtol=0)
    torch.testing.assert_close(b.cpu(), reference / dist.get_world_size(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grad_reduce_in_fp32": False},
        {"grad_reduce_in_fp32": True, "grad_reduce_in_fp64": True},
        {"grad_reduce_in_fp32": True, "num_distributed_optimizer_instances": 2},
        {"grad_reduce_in_fp32": True, "reduce_scatter_with_fp32_accumulation": True},
    ],
)
def test_unsupported_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError, match="grad_reduce_in_rank_order"):
        DistributedDataParallelConfig(grad_reduce_in_rank_order=True, **kwargs)


def test_wrong_wire_dtype_is_rejected():
    with pytest.raises(ValueError, match="FP32"):
        all_reduce_rank_ordered(
            torch.ones(4, device="cuda", dtype=torch.bfloat16),
            op=dist.ReduceOp.SUM,
            group=dist.group.WORLD,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_training_parser_enables_rank_order_without_fp64(enabled):
    parser = add_megatron_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--grad-reduce-in-rank-order"] if enabled else [])
    assert args.grad_reduce_in_rank_order is enabled
    assert args.grad_reduce_in_fp64 is False
    assert args.grad_norm_in_fp64 is False
