# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Union

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig


@dataclass
class TorchFullyShardedDataParallelConfig(DistributedDataParallelConfig):
    """Configuration for TorchFullyShardedDataParallel."""

    reshard_after_forward: Union[bool, int] = True
    """
    Controls the parameter behavior after forward.

    See PyTorch for complete documentation:
    https://github.com/pytorch/pytorch/blob/ac8ddf115065106f038865389a07f2d0c9ed5e11/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L97C31-L97C49 # pylint: disable=line-too-long 
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
