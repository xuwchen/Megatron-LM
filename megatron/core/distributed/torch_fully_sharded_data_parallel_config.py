# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Literal

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig


@dataclass
class TorchFullyShardedDataParallelConfig(DistributedDataParallelConfig):
    """Configuration for TorchFullyShardedDataParallel."""

    gradient_accumulation_mode: Literal["classic", "partial_reduce_scatter"] = "classic"
    """
    Controls communication for intermediate gradient-accumulation microbatches.
    ``classic`` disables both reduce-scatter and all-reduce, while
    ``partial_reduce_scatter`` keeps reduce-scatter enabled and defers the HSDP
    replica all-reduce until the final microbatch.
    Partial reduce-scatter requires every rank and microbatch to execute the same
    FSDP module groups in the same order; unused-parameter reduction only pads
    parameters inside groups that were actually executed.
    """

    reshard_after_forward: bool | int | None = None
    """
    Controls the parameter behavior after forward. ``None`` selects PyTorch's
    automatic policy: reshard non-root modules and keep the root unsharded.
    On PyTorch 2.6 and 2.7, which predate the explicit automatic policy, MCore
    passes ``True`` to preserve their equivalent root-special-casing behavior.

    See PyTorch for complete documentation:
    https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
    """

    reduce_scatter_unused_params: bool = False
    """
    Include zero gradients for locally unused parameters so rank-divergent
    conditional control flow still produces matching FSDP2 reduce-scatter collectives.
    Requires PyTorch FSDP2's set_reduce_scatter_unused_params API.
    This does not make rank-divergent calls to whole FSDP modules safe in forward.
    Parameters unused on every rank receive zero gradients instead of None, which may
    change optimizer state and weight-decay behavior.
    """

    clone_output_views: bool = False
    """
    Clone differentiable output views from the root module and language embedding
    before FSDP2 registers its pre-backward hooks. This prevents downstream
    in-place operations from silently dropping those hooks. This is opt-in because
    cloning breaks output aliasing and may add activation memory and copy overhead.
    """
