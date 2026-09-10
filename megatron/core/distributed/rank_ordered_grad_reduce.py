# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP32 gradient collectives with a fixed per-element process-group rank order.

All-to-all transfers contributions without arithmetic. A local FP32 left fold
reduces each shard, followed by all-gather for an all-reduce. This trades native
NCCL reduction kernels for explicit arithmetic and a full-size receive buffer.
Handles must be completed outside distributed coalescing managers.
"""

from typing import Any, Iterable

import torch

from megatron.core.utils import nvtx_decorator


class RankOrderedReductionWork:
    """Own FP32 communication buffers and finish one rank-ordered reduction."""

    def __init__(
        self,
        work: Any,
        source: torch.Tensor,
        received: torch.Tensor,
        destination: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        mean: bool,
        gather: bool,
    ) -> None:
        self.work = work
        self.source = source
        self.received = received
        self.destination = destination
        self.group = group
        self.mean = mean
        self.gather = gather

    @torch.no_grad()
    @nvtx_decorator(message="rank_ordered_grad_reduce.finish")
    def wait(self) -> None:
        """Complete FP32 arithmetic/copies on the caller's stream exactly once."""
        if self.destination is None:
            return
        if self.work is not None:
            self.work.wait()
        size = self.group.size()
        contributions = self.received.view(size, -1)
        reduced = contributions[0].clone()
        for rank in range(1, size):
            reduced.add_(contributions[rank])
        if self.mean:
            reduced.div_(size)
        if self.gather and size > 1:
            gathered = torch.empty_like(self.source)
            torch.distributed.all_gather_into_tensor(gathered, reduced, group=self.group)
        else:
            gathered = reduced
        self.destination.copy_(gathered[: self.destination.numel()].view_as(self.destination))
        if self.source.is_cuda:
            stream = torch.cuda.current_stream(self.source.device)
            for tensor in (self.source, self.received, reduced, gathered):
                tensor.record_stream(stream)
        self.work = self.source = self.received = self.destination = None


class RankOrderedReductionWorkGroup:
    """Complete all deferred rank-ordered collectives in their issue order."""

    def __init__(self, works: Iterable[RankOrderedReductionWork | None]) -> None:
        self.works = [work for work in works if work is not None]

    def wait(self) -> None:
        """Complete pending operations, allowing repeated waits."""
        for work in self.works:
            work.wait()
        self.works.clear()


def _validate(tensor, op, group):
    if tensor.dtype != torch.float32:
        raise ValueError('Rank-ordered gradient reduction requires FP32 buffers')
    if group is None:
        raise ValueError('Rank-ordered reduction requires an explicit process group')
    if op not in (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.AVG):
        raise ValueError('Rank-ordered reduction supports SUM and AVG only')
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Rank-ordered gradient reduction requires eager execution')


def _start(output, input, op, group, async_op, gather):
    _validate(input, op, group)
    _validate(output, op, group)
    if output.device != input.device:
        raise ValueError('Rank-ordered collective buffers must be on the same device')
    size = group.size()
    source = input.reshape(-1).contiguous()
    remainder = source.numel() % size
    if remainder:
        padded = torch.zeros(
            source.numel() + size - remainder, device=input.device, dtype=input.dtype
        )
        padded[: source.numel()].copy_(source)
        source = padded
    if size > 1:
        received = torch.empty_like(source)
        work = torch.distributed.all_to_all_single(received, source, group=group, async_op=True)
    else:
        received = source
        work = None
    handle = RankOrderedReductionWork(
        work, source, received, output, group, op == torch.distributed.ReduceOp.AVG, gather
    )
    if async_op:
        return handle
    handle.wait()
    return None


@torch.no_grad()
def all_reduce_rank_ordered(
    tensor: torch.Tensor,
    *,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool = False,
) -> RankOrderedReductionWork | None:
    """Reduce FP32 contributions in ascending group-rank order and replicate."""
    return _start(tensor, tensor, op, group, async_op, gather=True)


@torch.no_grad()
def reduce_scatter_rank_ordered(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool = False,
) -> RankOrderedReductionWork | None:
    """Reduce FP32 contributions in group-rank order and keep the local shard."""
    _validate(input, op, group)
    if input.numel() != output.numel() * group.size():
        raise ValueError('Rank-ordered reduce-scatter requires equal-sized output shards')
    return _start(output, input, op, group, async_op, gather=False)
