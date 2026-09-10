# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in FP64 communication for FP32 gradient buffers.

Keep local wgrad computation and optimizer storage unchanged, and round a
collective result once when copying it back to its FP32 destination. The staging
buffers increase communication bytes and temporary memory. These helpers must
not run inside a distributed coalescing manager: their handles own a deferred
copy in addition to the NCCL work.
"""

from typing import Any, Iterable

import torch


class GradientReductionWork:
    """Retain FP64 buffers until communication and the destination copy finish."""

    def __init__(
        self,
        work: Any,
        source: torch.Tensor,
        reduced: torch.Tensor,
        destination: torch.Tensor,
        divisor: int,
    ) -> None:
        self.work = work
        self.source = source
        self.reduced = reduced
        self.destination = destination
        self.divisor = divisor

    @torch.no_grad()
    def wait(self) -> None:
        """Wait on the caller's stream and perform the FP32 roundback once."""
        if self.destination is None:
            return
        self.work.wait()
        if self.divisor != 1:
            self.reduced.div_(self.divisor)
        self.destination.copy_(self.reduced)
        if self.reduced.is_cuda:
            stream = torch.cuda.current_stream(self.reduced.device)
            # The wait/copy may run on a different stream from allocation.
            self.source.record_stream(stream)
            self.reduced.record_stream(stream)
        self.work = self.source = self.reduced = self.destination = None


class GradientReductionWorkGroup:
    """Wait on every deferred gradient copy in a bucket group."""

    def __init__(self, works: Iterable[GradientReductionWork | None]) -> None:
        self.works = [work for work in works if work is not None]

    def wait(self) -> None:
        """Complete all queued gradient copies, safely allowing repeated waits."""
        for work in self.works:
            work.wait()
        self.works.clear()


def _validate(tensor, op, group):
    if tensor.dtype != torch.float32:
        raise ValueError('FP64 gradient reduction requires FP32 gradient buffers')
    if op not in (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.AVG):
        raise ValueError('FP64 gradient reduction supports SUM and AVG only')
    if group is None:
        raise ValueError('FP64 gradient reduction requires an explicit process group')
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('FP64 gradient reduction does not support CUDA graph capture')


@torch.no_grad()
def all_reduce_fp64(
    tensor: torch.Tensor,
    *,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool = False,
) -> GradientReductionWork | None:
    """Reduce an FP32 buffer using FP64 SUM, optionally dividing before roundback."""
    _validate(tensor, op, group)
    staged = tensor.to(torch.float64)
    work = torch.distributed.all_reduce(
        staged, op=torch.distributed.ReduceOp.SUM, group=group, async_op=True
    )
    handle = GradientReductionWork(
        work, staged, staged, tensor, group.size() if op == torch.distributed.ReduceOp.AVG else 1
    )
    if async_op:
        return handle
    handle.wait()
    return None


@torch.no_grad()
def reduce_scatter_fp64(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool = False,
) -> GradientReductionWork | None:
    """Reduce-scatter FP32 gradients with FP64 staging and a deferred FP32 copy."""
    _validate(input, op, group)
    _validate(output, op, group)
    if input.numel() != output.numel() * group.size():
        raise ValueError('FP64 reduce-scatter requires equal-sized output shards')
    staged = input.to(torch.float64)
    reduced = torch.empty_like(output, dtype=torch.float64)
    work = torch.distributed.reduce_scatter_tensor(
        reduced, staged, op=torch.distributed.ReduceOp.SUM, group=group, async_op=True
    )
    handle = GradientReductionWork(
        work, staged, reduced, output, group.size() if op == torch.distributed.ReduceOp.AVG else 1
    )
    if async_op:
        return handle
    handle.wait()
    return None
