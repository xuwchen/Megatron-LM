# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layout-independent FP32 L2 reduction over named logical parameter blocks.

Only completed block powers and raw split-block fragments are communicated.
Every multiplication, addition and square root is FP32. Explicit binary trees
avoid a fused local norm's dependence on tensor sizes and ownership boundaries.
This optional eager path trades extra kernels and communication for a fixed
arithmetic order; it is not the default fused norm implementation.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch

BLOCK_SIZE = 1024
METADATA_ATTR = '_fixed_grad_norm_metadata'
_PLAN_CACHE: dict[tuple, Any] = {}


@dataclass(frozen=True)
class FixedGradNormMetadata:
    """Identify a parameter shard by its logical name, offset and total size."""

    name: str
    offset: int
    numel: int

    def narrow(self, start: int) -> 'FixedGradNormMetadata':
        """Return metadata for a view starting at a local element offset."""
        return replace(self, offset=self.offset + start)


def configure_fixed_grad_norm_metadata(model_chunks: list, pg_collection: Any) -> None:
    """Name TP1/PP1 parameters and record GTP offsets before optimizer sharding."""
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam

    if pg_collection.tp.size() != 1 or pg_collection.pp.size() != 1:
        raise ValueError('Fixed-order gradient norms currently require TP1 and PP1')
    expert_rank = pg_collection.ep.rank()
    for chunk_index, model in enumerate(model_chunks):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            while name.startswith('module.'):
                name = name[len('module.') :]
            name = f'{chunk_index}:{name}'
            if not getattr(param, 'allreduce', True):
                name += f':ep{expert_rank}'
            if isinstance(param, GTPShardedParam):
                offset = param.group.rank() * param.numel()
                row_size = param.numel() // param.shape[0]
                numel = param.numel() * param.group.size() - param.pad_length * row_size
            else:
                offset, numel = 0, param.numel()
            setattr(param, METADATA_ATTR, FixedGradNormMetadata(name, offset, numel))


def copy_fixed_grad_norm_metadata(
    destination: torch.Tensor, source: torch.Tensor, start: int = 0
) -> None:
    """Preserve logical element positions when creating optimizer copies/views."""
    if hasattr(source, METADATA_ATTR):
        setattr(destination, METADATA_ATTR, getattr(source, METADATA_ATTR).narrow(start))


def _sum_tree(rows: torch.Tensor) -> torch.Tensor:
    # Each kernel performs exactly one FP32 addition per output. The number of
    # rows never changes the arithmetic tree within a row.
    while rows.shape[-1] > 1:
        rows = rows[..., 0::2] + rows[..., 1::2]
    return rows.reshape(-1)


@dataclass
class _Piece:
    grad_index: int
    local_start: int
    local_end: int
    payload_start: int
    payload_end: int
    global_block: int
    fragment_offset: int | None = None


class _Plan:
    def __init__(self, descriptions: list[list[tuple]]) -> None:
        sizes, ownership = {}, {}
        for records in descriptions:
            for name, offset, numel, length in records:
                if min(offset, numel, length) < 0:
                    raise ValueError('Fixed-order norm shard sizes and offsets must be nonnegative')
                if sizes.setdefault(name, numel) != numel:
                    raise ValueError(f'Inconsistent logical gradient size: {name}')
                start, end = min(offset, numel), min(offset + length, numel)
                if end > start:
                    ownership.setdefault(name, []).append((start, end))
        for name, numel in sizes.items():
            cursor = 0
            for start, end in sorted(ownership.get(name, [])):
                if start != cursor:
                    raise ValueError(
                        f'Missing or overlapping gradient ownership: {name} at {cursor}'
                    )
                cursor = end
            if cursor != numel:
                raise ValueError(
                    f'Incomplete gradient ownership: {name} ends at {cursor}, expected {numel}'
                )
        bases, count = {}, 0
        for name in sorted(sizes):
            bases[name] = count
            count += (sizes[name] + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.num_blocks = count
        self.pieces, self.counts = [], []
        fragment_blocks = set()
        for records in descriptions:
            pieces, cursor = [], 0
            for index, (name, offset, numel, length) in enumerate(records):
                start, end = min(offset, numel), min(offset + length, numel)
                full_start = min(end, (start + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE)
                full_end = max(full_start, end // BLOCK_SIZE * BLOCK_SIZE)
                if full_end > full_start:
                    nblocks = (full_end - full_start) // BLOCK_SIZE
                    pieces.append(
                        _Piece(
                            index,
                            full_start - offset,
                            full_end - offset,
                            cursor,
                            cursor + nblocks,
                            bases[name] + full_start // BLOCK_SIZE,
                        )
                    )
                    cursor += nblocks
                for begin, finish in ((start, full_start), (full_end, end)):
                    if begin == finish:
                        continue
                    block = bases[name] + begin // BLOCK_SIZE
                    nvalues = finish - begin
                    pieces.append(
                        _Piece(
                            index,
                            begin - offset,
                            finish - offset,
                            cursor,
                            cursor + nvalues,
                            block,
                            begin % BLOCK_SIZE,
                        )
                    )
                    fragment_blocks.add(block)
                    cursor += nvalues
            self.pieces.append(pieces)
            self.counts.append(cursor)
        self.fragment_blocks = sorted(fragment_blocks)
        self.fragment_rows = {block: index for index, block in enumerate(self.fragment_blocks)}
        self.max_count = max(self.counts, default=0)


def _get_plan(grads: list[torch.Tensor], group: Any, device: torch.device) -> _Plan:
    descriptions = []
    for grad in grads:
        if not hasattr(grad, METADATA_ATTR):
            raise ValueError('Fixed-order gradient norm requires logical parameter metadata')
        meta = getattr(grad, METADATA_ATTR)
        descriptions.append((meta.name, meta.offset, meta.numel, grad.numel()))
    key = (group, device, tuple(descriptions))
    plan = _PLAN_CACHE.get(key)
    # All ranks must take the same cache-hit/miss branch, including when one
    # rank's gradient set changes or its bounded cache has evicted a plan.
    cached = torch.tensor([plan is not None], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(cached, op=torch.distributed.ReduceOp.MIN, group=group)
    if not cached.item():
        gathered = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(gathered, descriptions, group=group)
        plan = _Plan(gathered)
        if len(_PLAN_CACHE) >= 32:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = plan
    return plan


@torch.no_grad()
def get_fixed_grad_norm_fp32(grads: list[torch.Tensor], group: Any = None) -> torch.Tensor:
    """Compute a canonical FP32 L2 norm across arbitrary named parameter shards.

    Args:
        grads: Unique optimizer-owned gradients carrying FixedGradNormMetadata.
        group: Existing gradient-statistics process group; None denotes WORLD.

    Returns:
        One FP32 CUDA element, suitable for the existing tensor clipping kernel.
    """
    device = torch.device('cuda', torch.cuda.current_device())
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Fixed-order gradient norms require eager execution')
    if any(grad.dtype != torch.float32 or grad.device != device for grad in grads):
        raise ValueError(
            'Fixed-order gradient norms require FP32 gradients on the current CUDA device'
        )
    plan = _get_plan(grads, group, device)
    if not plan.num_blocks:
        return torch.zeros(1, dtype=torch.float32, device=device)
    rank = torch.distributed.get_rank(group)
    payload = torch.zeros(plan.max_count, dtype=torch.float32, device=device)
    for piece in plan.pieces[rank]:
        values = grads[piece.grad_index].reshape(-1)[piece.local_start : piece.local_end]
        if piece.fragment_offset is None:
            values = _sum_tree(values.reshape(-1, BLOCK_SIZE).square())
        payload[piece.payload_start : piece.payload_end].copy_(values)
    gathered = torch.empty(
        torch.distributed.get_world_size(group) * plan.max_count, dtype=torch.float32, device=device
    )
    torch.distributed.all_gather_into_tensor(gathered, payload, group=group)
    gathered = gathered.reshape(-1, plan.max_count)
    powers = torch.zeros(plan.num_blocks, dtype=torch.float32, device=device)
    fragments = torch.zeros(
        (len(plan.fragment_blocks), BLOCK_SIZE), dtype=torch.float32, device=device
    )
    for source_rank, pieces in enumerate(plan.pieces):
        for piece in pieces:
            values = gathered[source_rank, piece.payload_start : piece.payload_end]
            if piece.fragment_offset is None:
                powers[piece.global_block : piece.global_block + values.numel()].copy_(values)
            else:
                row = plan.fragment_rows[piece.global_block]
                begin = piece.fragment_offset
                fragments[row, begin : begin + values.numel()].copy_(values)
    if plan.fragment_blocks:
        indices = torch.tensor(plan.fragment_blocks, dtype=torch.long, device=device)
        powers[indices] = _sum_tree(fragments.square())
    padded = torch.zeros(
        1 << (plan.num_blocks - 1).bit_length(), dtype=torch.float32, device=device
    )
    padded[: plan.num_blocks].copy_(powers)
    return _sum_tree(padded).sqrt()
