# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from typing import Optional

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.transformer.utils import cat_with_oom_fallback


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: list[int], split_names: list[str], split_dim: int
) -> ShardedTensorFactory:
    """Split a fused tensor into logical checkpoint tensors.

    Torch FSDP2 may shard the TP-local fused tensor across data-parallel ranks. In
    that case one physical shard can overlap multiple logical sections, so the
    split must use global checkpoint metadata instead of the local tensor shape.
    """
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    split_axis = split_dim + orig_sh_ten_no_data.prepend_axis_num
    tp_local_size = sum(split_sections)
    global_axis_size = orig_sh_ten_no_data.global_shape[split_axis]
    if global_axis_size % tp_local_size != 0:
        raise ValueError(
            f"Global dimension must contain whole TP-local fused tensors, got "
            f"{split_sections=} vs {global_axis_size=}"
        )
    tp_size = global_axis_size // tp_local_size

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        local_axis_size = factory_sh_ten.local_shape[split_dim]
        global_axis_offset = factory_sh_ten.global_offset[split_axis]
        tp_rank, local_start = divmod(global_axis_offset, tp_local_size)
        local_stop = local_start + local_axis_size
        if tp_rank >= tp_size or local_stop > tp_local_size:
            raise ValueError(
                "A physical shard must fit in one TP-local fused tensor, got "
                f"{global_axis_offset=}, {local_axis_size=}, {tp_local_size=}, {tp_size=}"
            )

        # Preserve the regular-grid representation when FSDP has not split
        # the TP-local tensor. This keeps legacy checkpoint metadata stable.
        if local_start == 0 and local_axis_size == tp_local_size:
            chunk_sh_tens = []
            split_start = 0
            for split_size, split_name in zip(split_sections, split_names):
                split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
                for sh_ten in split_chunks:
                    sh_ten.key = f"{sh_ten.key}.{split_name}"
                chunk_sh_tens.extend(split_chunks)
                split_start += split_size
            return chunk_sh_tens

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_stop = split_start + split_size
            overlap_start = max(local_start, split_start)
            overlap_stop = min(local_stop, split_stop)
            if overlap_start < overlap_stop:
                overlap_size = overlap_stop - overlap_start
                chunk_data = t.narrow(split_dim, overlap_start - local_start, overlap_size)
                chunk_global_shape = list(factory_sh_ten.global_shape)
                chunk_global_shape[split_axis] = split_size * tp_size
                chunk_global_offset = list(factory_sh_ten.global_offset)
                chunk_global_offset[split_axis] = tp_rank * split_size + overlap_start - split_start
                chunk_sh_tens.append(
                    replace(
                        factory_sh_ten,
                        key=f"{key}.{split_name}",
                        data=chunk_data,
                        local_shape=tuple(chunk_data.shape),
                        global_shape=tuple(chunk_global_shape),
                        global_offset=tuple(chunk_global_offset),
                        axis_fragmentations=None,
                    )
                )
            split_start += split_size

        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    factory = ShardedTensorFactory(
        orig_sh_ten.key,
        orig_sh_ten.data,
        sh_ten_build_fn,
        cat_with_oom_fallback,
        orig_sh_ten.replica_id,
        flattened_range=orig_sh_ten.flattened_range,
    )
    if getattr(orig_sh_ten, "is_data_parallel_fully_shard", False):
        factory.is_data_parallel_fully_shard = True
    return factory
