# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import logging
import os
from contextlib import nullcontext
from functools import lru_cache

import torch

from megatron.core.utils import is_torch_min_version, log_single_rank

logger = logging.getLogger(__name__)

# MCore NCCL allocator uses the NCCL allocator exposed by ProcessGroupNCCL.
# This avoids runtime inline C++ compilation while keeping allocations backed
# by NCCL's user-buffer allocator.


def _resolve_cuda_device(device=None):
    """Return an indexed CUDA device for ProcessGroupNCCL backend lookup."""
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL allocator requires CUDA, but CUDA is not available.")
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, str):
        device = torch.device(device)
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be None, int, str, or torch.device, got {type(device)}")
    if device.type != "cuda":
        raise RuntimeError(f"NCCL allocator only supports CUDA devices, got {device}.")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _get_default_group():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "NCCL allocator requires a process group. Pass group=... or initialize "
            "torch.distributed before calling create_nccl_mem_pool()."
        )
    return torch.distributed.distributed_c10d._get_default_group()


def _get_nccl_backend(group=None, device=None):
    device = _resolve_cuda_device(device)
    group = _get_default_group() if group is None else group
    if not hasattr(group, "_get_backend"):
        raise TypeError(f"group must be a torch.distributed.ProcessGroup, got {type(group)}")
    return group._get_backend(device), group, device


def get_nccl_allocator(group=None, device=None):
    """Return the NCCL user-buffer allocator exposed by ProcessGroupNCCL."""
    backend, group, device = _get_nccl_backend(group=group, device=device)
    allocator = getattr(backend, "mem_allocator", None)
    if allocator is None:
        group_desc = getattr(group, "group_desc", None)
        raise RuntimeError(
            "ProcessGroupNCCL backend does not expose mem_allocator on this PyTorch build. "
            "Megatron NCCL UBR cannot run without the legacy inline C++ allocator. "
            f"group={group!r}, group_desc={group_desc}, device={device}."
        )

    # Some downstream builds expose a zero-argument method instead of a property.
    if callable(allocator):
        allocator = allocator()
    return allocator


@lru_cache(maxsize=None)
def get_func_args(func):
    """
    Get the argument names of a function.
    """
    import inspect

    sig = inspect.signature(func)
    return [arg.name for arg in sig.parameters.values()]


def create_nccl_mem_pool(
    symmetric=None, group=None, device=None
):  # symmetric: bool | None = None -> torch.cuda.MemPool:
    """
    Create a memory pool using the NCCL allocator.
    """
    if not is_torch_min_version("2.9.0a0") and symmetric is True:
        logging.info(
            f"Symmetric memory pool is not supported with torch version < 2.9.0a0"
            f"Current torch version: {torch.__version__}"
            "falling back to non-symmetric memory pool"
        )
        symmetric = False

    allocator = get_nccl_allocator(group=group, device=device)
    if not symmetric:
        _pool = torch.cuda.MemPool(allocator)
    else:
        if 'symmetric' in get_func_args(torch.cuda.MemPool):
            # The PyTorch version >= 2.9.0a0 and before PyTorch PR #161238,
            # The symmetric knob should passed to the MemPool constructor.
            # Since PyTorch PR #161238 symmetric knob is now in registration function.
            _pool = torch.cuda.MemPool(allocator, symmetric=symmetric)
        elif 'symm_mem' in get_func_args(torch.cuda.MemPool):
            # This path handles argument name divergence between
            # nvidia pytorch and the official pytorch.
            _pool = torch.cuda.MemPool(allocator, symm_mem=symmetric)
        else:
            # This path handles the case where the symmetric knob is in the registration function.
            _pool = torch.cuda.MemPool(allocator)
    return _pool


def init() -> None:
    """
    Initialize the NCCL allocator.

    PyTorch tracks memory registration at the pool level, not per allocation.
    If a pool already contains allocations from a previous context, attempting
    to register it again will re-register all existing allocations and may
    trigger NCCL errors. To avoid this, the pool is explicitly deregistered
    on entry and re-registered on exit for each context use.
    """
    # Enable NCCL NVLS by default, while preserving an explicit user or recipe setting.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
    # Disables the use of the tensor register allocator hook
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"
    log_single_rank(
        logger,
        logging.INFO,
        "[MCORE][NCCL_ALLOCATOR] Configured ProcessGroupNCCL mem_allocator",
    )


# register_mem_pool/deregister_mem_pool are used for manual (de)registration of the memory pool.
# They are used in the case of FSDP manual registration.
def register_mem_pool(pool, group, symmetric=True):
    """
    Register a memory pool to a group.
    symmetric: bool, this is for future use.
    """
    backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
    if symmetric:
        try:
            backend.register_mem_pool(pool, symm=symmetric)
        except TypeError:
            # Older PyTorch/APIs without 'symm' keyword.
            log_single_rank(
                logger,
                logging.WARNING,
                "[MCORE][NCCL_ALLOCATOR] Failed in symmetric registration. "
                "Falling back to registration api without 'symm' keyword!!",
            )
            backend.register_mem_pool(pool)
    else:
        backend.register_mem_pool(pool)


def deregister_mem_pool(pool, group):
    """
    Deregister a memory pool from a group.
    """
    backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
    if pool.snapshot():
        backend.deregister_mem_pool(pool)


# Preserve the original APEX NCCL allocator interface for backward compatibility
class nccl_mem:
    """
    An NCCL memory allocator, which inherits APEX nccl_allocator implementation.
    """

    def __init__(self, pool, enabled=True, device=None, group=None, symmetric=True):
        self.device = None
        self.group = None
        self.mem_context = None
        self.pool = pool
        self.symmetric = symmetric

        if enabled:
            if device is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, int):
                self.device = torch.device("cuda", device)
            elif isinstance(device, str):
                assert "cuda" in device, "only cuda devices are supported"
                self.device = torch.device(device)

            if group is None:
                self.group = torch.distributed.distributed_c10d._get_default_group()
            else:
                self.group = group

            self.mem_context = torch.cuda.use_mem_pool(self.pool)
        else:
            self.mem_context = nullcontext()

    def __enter__(self):
        self.mem_context.__enter__()
        if self.group is not None:
            # If the pool is not empty, deregister the pool from the group.
            if self.pool.snapshot():
                backend = self.group._get_backend(self.device)
                try:
                    # Deregister first to avoid duplicate registration of previously
                    # registered memory.
                    backend.deregister_mem_pool(self.pool)
                except RuntimeError:
                    desc = getattr(self.group, "group_desc", None)
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        f"[MCORE][NCCL_ALLOCATOR] Failed to deregister mem pool from"
                        f"{repr(self.group)}({desc}) group!!",
                    )

    def __exit__(self, *args):
        if self.group is not None:
            backend = self.group._get_backend(self.device)
            try:
                # Prefer attempting symmetric registration first; fall back if unsupported.
                if self.symmetric:
                    try:
                        # Since PyTorch PR #161238 symmetric knob is now in registration function.
                        backend.register_mem_pool(self.pool, symm=self.symmetric)
                    except TypeError:
                        # Older PyTorch/APIs without 'symm' keyword.
                        log_single_rank(
                            logger,
                            logging.WARNING,
                            "[MCORE][NCCL_ALLOCATOR] Failed in symmetric registration. "
                            "Falling back to non-symmetric registration!!",
                        )
                        backend.register_mem_pool(self.pool)
                else:
                    backend.register_mem_pool(self.pool)
            except RuntimeError:
                desc = getattr(self.group, "group_desc", None)
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"[MCORE][NCCL_ALLOCATOR] Failed to register mem pool to"
                    f"{repr(self.group)}({desc}) group!!",
                )

        self.mem_context.__exit__(*args)


class MultiGroupMemPoolAllocator:
    """
    A custom allocator class that registers a single memory pool with multiple communication groups.

    Use cases:
    - [FSDP+EP] In case of FSDP with EP, expert layer (expert-dp) and non-expert layer (dp) use
      different communicator groups. The same memory pool has to be registered to both the groups.
    - [Hybrid FSDP/DP] In case of Hybrid FSDP/DP, there are inter-dp group and intra-dp group.
      The same memory pool has to be registered to both the groups.
    - [Hybrid FSDP/DP + EP] In case of Hybrid FSDP/DP + EP, there are inter-dp, intra-dp, and
      expert-dp groups. The same memory pool has to be registered to all the groups.

    Example:
        ```
        import megatron.core.nccl_allocator as nccl_allocator
        nccl_allocator.init()
        group_1 = torch.distributed.new_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7], backend="nccl")
        group_2 = torch.distributed.new_group(ranks=[0, 2, 4, 6], backend="nccl")
        pool = nccl_allocator.create_nccl_mem_pool(group=group_1)
        with MultiGroupMemPoolAllocator(pool, [group_1, group_2]):
            a = torch.zeros(1024, dtype=torch.float32, device="cuda")
            b = torch.zeros(1024, dtype=torch.float32, device="cuda")
        ```
    """

    def __init__(
        self, pool, groups, symmetric=True
    ):  # pool: torch.cuda.MemPool, groups: List[torch.distributed.ProcessGroup]
        self.pool = pool
        self.groups = groups
        self.mem_context = torch.cuda.use_mem_pool(self.pool)
        self.symmetric = symmetric

        assert isinstance(self.pool, torch.cuda.MemPool), "pool must be a torch.cuda.MemPool"
        assert isinstance(self.groups, list), "groups must be a list"
        assert all(
            isinstance(group, torch.distributed.ProcessGroup) for group in self.groups
        ), "groups must be a list of torch.distributed.ProcessGroup"

    def __enter__(self):
        self.mem_context.__enter__()
        # If the pool is not empty, deregister the pool from all the groups.
        if self.pool.snapshot():
            for group in self.groups:
                backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
                try:
                    # Since the registration is done in mempool granularity, we need to deregister
                    # the tensors in the mempool and re-register the mempool including
                    # the newly created tensors after the context is exited.
                    backend.deregister_mem_pool(self.pool)
                except RuntimeError:
                    desc = getattr(group, "group_desc", None)
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        f"[MCORE][MultiGroupMemPoolAllocator] Failed to deregister mem pool from"
                        f"{repr(group)}({desc}) group!!",
                    )

    def __exit__(self, *args):
        for group in self.groups:
            backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
            try:
                # Prefer attempting symmetric registration first; fall back if unsupported.
                if self.symmetric:
                    try:
                        # Since PyTorch PR #161238 symmetric knob is now in registration function.
                        backend.register_mem_pool(self.pool, symm=self.symmetric)
                    except TypeError:
                        # Older PyTorch/APIs without 'symm' keyword.
                        log_single_rank(
                            logger,
                            logging.WARNING,
                            "[MCORE][MultiGroupMemPoolAllocator] "
                            "Failed in symmetric registration. "
                            "Falling back to non-symmetric registration!!",
                        )
                        backend.register_mem_pool(self.pool)
                else:
                    backend.register_mem_pool(self.pool)
            except RuntimeError:
                desc = getattr(group, "group_desc", None)
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"[MCORE][MultiGroupMemPoolAllocator] Failed to register mem pool to"
                    f"{repr(group)}({desc}) group!!",
                )
        self.mem_context.__exit__(*args)


class MemPoolAllocatorWithoutRegistration:
    """
    An allocator class that uses allocates memory without registering to any communication group.
    Users are expected to register the memory manually to the communication groups.
    """

    def __init__(self, pool):
        self.pool = pool
        self.mem_context = torch.cuda.use_mem_pool(self.pool)

    def __enter__(self):
        self.mem_context.__enter__()

    def __exit__(self, *args):
        self.mem_context.__exit__(*args)
