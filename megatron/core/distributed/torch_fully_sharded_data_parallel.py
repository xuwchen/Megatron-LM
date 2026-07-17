# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Set, Tuple, Type, Union

import torch
from torch.distributed.utils import _apply_to_tensors

try:
    from torch.distributed import DeviceMesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    HAVE_FSDP = True
except ImportError:
    HAVE_FSDP = False

from torch.distributed import ProcessGroup

from megatron.core.fp8_utils import is_float8tensor

from .. import parallel_state, tensor_parallel
from ..models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from ..models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from ..process_groups_config import ProcessGroupCollection
from ..transformer.transformer_config import TransformerConfig
from ..transformer.transformer_layer import TransformerLayer
from ..utils import is_torch_min_version
from .data_parallel_base import _BaseDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig


def _build_fsdp_wrap_plan(
    root_module: torch.nn.Module, sub_modules_to_wrap: Set[Type[torch.nn.Module]]
) -> Tuple[List[torch.nn.Module], List[torch.nn.Module]]:
    """Build stable forward and bottom-up orders for FSDP2 module wrapping."""
    forward_order = []
    bottom_up_order = []
    visited_module_ids = set()

    def visit(module: torch.nn.Module) -> None:
        module_id = id(module)
        if module_id in visited_module_ids:
            return
        visited_module_ids.add(module_id)

        should_wrap = module is not root_module and any(
            isinstance(module, sub_module_type) for sub_module_type in sub_modules_to_wrap
        )
        if should_wrap:
            forward_order.append(module)

        for child_module in module.children():
            visit(child_module)

        if should_wrap:
            bottom_up_order.append(module)

    visit(root_module)
    return forward_order, bottom_up_order


def _configure_fsdp_sum_gradient_reduction(module: torch.nn.Module) -> None:
    """Configure true SUM reduction without using BF16 NCCL PREMUL_SUM."""
    module.set_force_sum_reduction_for_comms(True)
    if setter := getattr(module, "set_gradient_divide_factor", None):
        setter(1.0)
    else:
        module.set_reduce_scatter_divide_factor(1.0)


def _resolve_fsdp_reshard_after_forward(
    reshard_after_forward: bool | int | None,
) -> bool | int | None:
    """Resolve the automatic reshard policy across supported PyTorch versions.

    PyTorch 2.8 introduced ``None`` for the automatic policy. PyTorch 2.6 and
    2.7 accept only bool or int, but passing ``True`` already keeps the root
    unsharded, which is equivalent to the newer automatic behavior.
    """
    if reshard_after_forward is None and not is_torch_min_version("2.8.0"):
        return True
    return reshard_after_forward


def _clone_fsdp_output_views(
    _module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any
) -> Any:
    """Clone grad-requiring output views before FSDP registers backward hooks.

    PyTorch FSDP2 attaches its pre-backward hook directly to module outputs.
    A downstream in-place operation on a view can silently replace that hook,
    skipping the parameter all-gather in backward. This hook is registered
    before fully_shard so FSDP observes independent tensors instead.
    """

    needs_clone = False

    def detect_view(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal needs_clone
        needs_clone |= tensor.requires_grad and tensor._base is not None
        return tensor

    _apply_to_tensors(detect_view, output)
    if not needs_clone:
        return output

    cloned_views = {}

    def clone_view(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.requires_grad or tensor._base is None:
            return tensor
        tensor_id = id(tensor)
        if tensor_id not in cloned_views:
            cloned_views[tensor_id] = tensor.clone()
        return cloned_views[tensor_id]

    return _apply_to_tensors(clone_view, output)


def _build_fsdp_device_mesh(
    process_group: ProcessGroup,
    pg_collection: Optional[ProcessGroupCollection],
    num_distributed_optimizer_instances: int,
) -> "DeviceMesh":
    """Build and validate the FSDP2 data-parallel device mesh."""
    if num_distributed_optimizer_instances < 1:
        raise ValueError(
            "num_distributed_optimizer_instances must be at least 1, got "
            f"{num_distributed_optimizer_instances}."
        )
    if num_distributed_optimizer_instances == 1:
        return DeviceMesh.from_group(process_group, device_type="cuda")

    if pg_collection is None:
        raise ValueError(
            "Torch FSDP2 HSDP requires pg_collection with dp_cp, intra_dp_cp, "
            "and inter_dist_opt process groups."
        )
    shard_group = getattr(pg_collection, "intra_dp_cp", None)
    replicate_group = getattr(pg_collection, "inter_dist_opt", None)
    if shard_group is None or replicate_group is None:
        raise ValueError(
            "Torch FSDP2 HSDP requires both pg_collection.intra_dp_cp (shard) "
            "and pg_collection.inter_dist_opt (replicate)."
        )

    full_ranks = torch.distributed.get_process_group_ranks(process_group)
    shard_ranks = torch.distributed.get_process_group_ranks(shard_group)
    replicate_ranks = torch.distributed.get_process_group_ranks(replicate_group)
    replicate_size = len(replicate_ranks)
    shard_size = len(shard_ranks)

    if replicate_size != num_distributed_optimizer_instances:
        raise ValueError(
            "Torch FSDP2 HSDP replicate group size does not match "
            "num_distributed_optimizer_instances: "
            f"{replicate_size} != {num_distributed_optimizer_instances}."
        )
    if replicate_size * shard_size != len(full_ranks):
        raise ValueError(
            "Torch FSDP2 HSDP topology does not cover the full dp_cp group: "
            f"replicate_size ({replicate_size}) * shard_size ({shard_size}) != "
            f"dp_cp_size ({len(full_ranks)})."
        )
    if len(set(full_ranks)) != len(full_ranks):
        raise ValueError(f"Torch FSDP2 HSDP dp_cp ranks must be unique, got {full_ranks}.")

    global_rank = torch.distributed.get_rank()
    if full_ranks.count(global_rank) != 1:
        raise ValueError(
            f"Current rank {global_rank} must occur exactly once in dp_cp ranks {full_ranks}."
        )

    rank_index = full_ranks.index(global_rank)
    replica_index, shard_index = divmod(rank_index, shard_size)
    expected_shard_ranks = full_ranks[replica_index * shard_size : (replica_index + 1) * shard_size]
    expected_replicate_ranks = full_ranks[shard_index::shard_size]
    if shard_ranks != expected_shard_ranks:
        raise ValueError(
            "Torch FSDP2 HSDP shard group does not match the current dp_cp mesh row: "
            f"expected {expected_shard_ranks}, got {shard_ranks}."
        )
    if replicate_ranks != expected_replicate_ranks:
        raise ValueError(
            "Torch FSDP2 HSDP replicate group does not match the current dp_cp mesh column: "
            f"expected {expected_replicate_ranks}, got {replicate_ranks}."
        )

    mesh = [
        full_ranks[offset : offset + shard_size] for offset in range(0, len(full_ranks), shard_size)
    ]
    return DeviceMesh.from_group(
        [replicate_group, shard_group],
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("replicate", "shard"),
    )


class TorchFullyShardedDataParallel(_BaseDataParallel):
    """
    Enables fully sharded data parallelism by wrapping the given model with
    the PyTorch FSDP2 API:
    https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
    To utilize this class, PyTorch version >= 2.6.0 is required.

    Args:
        config: Transformer config object.
        ddp_config: TorchDistributedDataParallel config object.
        module: Underlying model.
        sub_modules_to_wrap: Set of sub_modules to shard with FSDP.
            Parameters within each sub_module will be all-gathered just-in-time.
            The default set includes the following submodules derived from the
            GPT model architecture:
                TransformerLayer (all Transformer layers)
                LanguageModelEmbedding (initial embedding layer)
                RotaryEmbedding  (initial RoPE layer)
                tensor_parallel.ColumnParallelLinear (final output layer)

            User can set _fsdp_modules attribute on submodules to set additional
            submodules to shard with FSDP.
        process_group: Optional ProcessGroup to use for distributed operations. This is a
            backward-compatible 1D FSDP path and cannot be combined with pg_collection.
        pg_collection: Optional ProcessGroupCollection used to build the data-parallel mesh.
            HSDP requires dp_cp, intra_dp_cp, and inter_dist_opt groups.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        sub_modules_to_wrap: Set[Type[torch.nn.Module]] = {
            TransformerLayer,
            LanguageModelEmbedding,
            RotaryEmbedding,
            tensor_parallel.ColumnParallelLinear,
        },
        disable_bucketing: bool = False,
        process_group: Optional[ProcessGroup] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        assert (
            HAVE_FSDP
        ), 'TorchFullyShardedDataParallel requires PyTorch >= 2.6.0 with FSDP 2 support.'

        super().__init__(config=config, module=module)

        if process_group is not None and pg_collection is not None:
            raise ValueError("Specify only one of process_group and pg_collection for Torch FSDP2.")
        if pg_collection is not None:
            self.process_group = getattr(pg_collection, "dp_cp", None)
            if self.process_group is None:
                raise ValueError("Torch FSDP2 requires pg_collection.dp_cp.")
        elif process_group is not None:
            self.process_group = process_group
        else:
            # Backward-compatible fallback for direct wrapper callers.
            self.process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

        self.num_distributed_optimizer_instances = ddp_config.num_distributed_optimizer_instances
        self.is_hsdp = self.num_distributed_optimizer_instances > 1
        self.device_mesh = _build_fsdp_device_mesh(
            self.process_group, pg_collection, self.num_distributed_optimizer_instances
        )
        self.shard_process_group = pg_collection.intra_dp_cp if self.is_hsdp else self.process_group
        self.replicate_process_group = pg_collection.inter_dist_opt if self.is_hsdp else None
        reshard_after_forward = _resolve_fsdp_reshard_after_forward(
            getattr(ddp_config, "reshard_after_forward", None)
        )
        kwargs = {"mesh": self.device_mesh, "reshard_after_forward": reshard_after_forward}

        self.ddp_config = ddp_config
        self._no_sync_depth = 0
        self._gradient_scale_correction = 1.0

        def save_custom_attrs(module):
            custom_attrs = {}
            for name, param in module.named_parameters():
                attrs = vars(param)
                if is_float8tensor(param):
                    # disable fp8 transpose cache and perform transposing fp8 weights
                    # at each micro-batch because torch-FSDP doesn't recognize the
                    # micro-batch id, thus removing unnecessary memory stores
                    attrs['_fp8_attrs']['transpose_invalid'] = False
                    del attrs['_fp8_attrs']['transpose']
                custom_attrs[name] = {k: v for k, v in attrs.items()}
            return custom_attrs

        def restore_custom_attrs(module, custom_attrs):
            for name, param in module.named_parameters():
                if name in custom_attrs:
                    for attr_name, attr_value in custom_attrs[name].items():
                        setattr(param, attr_name, attr_value)

        # Save the custom attributes on Parameters before FSDP overwrites them.
        # See https://github.com/pytorch/pytorch/issues/136929.
        attrs = save_custom_attrs(self.module)

        # Local transformer implementation does not support ColumnParallelLinear.
        if config.transformer_impl == "local":
            sub_modules_to_wrap = [
                sub_module
                for sub_module in sub_modules_to_wrap
                if sub_module != tensor_parallel.ColumnParallelLinear
            ]
        sub_modules_to_wrap = set(sub_modules_to_wrap)
        for sub_module in self.module.modules():
            fsdp_modules = getattr(sub_module, "_fsdp_modules", [])
            for f in fsdp_modules:
                sub_modules_to_wrap.add(f)

        forward_order, bottom_up_order = _build_fsdp_wrap_plan(self.module, sub_modules_to_wrap)
        if getattr(ddp_config, "clone_output_views", False):
            for fsdp_module in [*bottom_up_order, self.module]:
                if fsdp_module is self.module or isinstance(fsdp_module, LanguageModelEmbedding):
                    # Register before fully_shard so this hook runs before FSDP's
                    # post-forward hook and protects its pre-backward hook.
                    fsdp_module.register_forward_hook(_clone_fsdp_output_views)
        for sub_module in bottom_up_order:
            # Wrap individual submodules to fetch parameters just-in-time rather than
            # conservatively fetching all parameters at the start of each iteration.
            # See https://github.com/pytorch/pytorch/issues/114299.
            fully_shard(sub_module, **kwargs)

        if config.recompute_granularity is not None:
            prev_module = None
            for sub_module in forward_order:
                # Explicitly set the FSDP backward prefetch schedule to prevent activation
                # recomputation from disrupting the automatically generated default schedule.
                sub_module.set_modules_to_backward_prefetch(
                    [prev_module] if prev_module is not None else []
                )
                prev_module = sub_module

        # Wrap the root module as required by the FSDP API.
        # See https://github.com/pytorch/pytorch/issues/114299.
        fully_shard(self.module, **kwargs)

        if config.calculate_per_token_loss:
            fsdp_modules = [*bottom_up_order, self.module]
            if all(
                getattr(fsdp_module, "set_force_sum_reduction_for_comms", None) is not None
                for fsdp_module in fsdp_modules
            ):
                # The setters are not recursive. Force true SUM for every communication
                # group since divide-factor 1 alone uses NCCL PREMUL_SUM, which is unsafe
                # for BF16 on older NCCL/PyTorch combinations.
                for fsdp_module in fsdp_modules:
                    _configure_fsdp_sum_gradient_reduction(fsdp_module)
            else:
                # PyTorch 2.6/2.7 cannot force pure SUM. Keep its default AVG and undo
                # the DP division when applying the final global-token normalization.
                self._gradient_scale_correction = float(self.process_group.size())

        if getattr(ddp_config, "reduce_scatter_unused_params", False):
            set_reduce_scatter_unused_params = getattr(
                self.module, "set_reduce_scatter_unused_params", None
            )
            if set_reduce_scatter_unused_params is None:
                raise RuntimeError(
                    "Torch FSDP2 unused-parameter reduction requires "
                    "FSDPModule.set_reduce_scatter_unused_params (PyTorch >= 2.13)."
                )
            set_reduce_scatter_unused_params(True, recurse=True)

        restore_custom_attrs(self.module, attrs)

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Disable FSDP2 gradient synchronization for intermediate microbatches.

        This context restores only FSDP2 synchronization controls. If a forward or
        backward raises, callers must discard the accumulated gradients, reset the
        root FSDP module's iteration state, and restart the accumulation cycle.
        """
        is_outermost = self._no_sync_depth == 0
        self._no_sync_depth += 1
        set_is_last_backward = getattr(self.module, "set_is_last_backward", None)
        try:
            if is_outermost:
                if set_is_last_backward is not None:
                    set_is_last_backward(False)
                self.module.set_requires_gradient_sync(False, recurse=True)
            yield
        finally:
            self._no_sync_depth -= 1
            if is_outermost:
                try:
                    self.module.set_requires_gradient_sync(True, recurse=True)
                finally:
                    if set_is_last_backward is not None:
                        # The next backward runs outside this context and completes reductions,
                        # waits for backward-prefetch communication, and clears iteration state.
                        set_is_last_backward(True)

    def scale_gradients(self, scaling_factor: Union[float, torch.Tensor]) -> None:
        """Scale each local FSDP2 gradient shard by the given factor."""
        gradient_scale_correction = getattr(self, "_gradient_scale_correction", 1.0)
        if gradient_scale_correction != 1.0:
            scaling_factor = scaling_factor * gradient_scale_correction

        with torch.no_grad():
            for param in self.module.parameters():
                grad = param.grad
                if grad is None:
                    continue
                if isinstance(grad, DTensor):
                    grad = grad.to_local()
                grad.mul_(scaling_factor)

    def finish_grad_sync(self, force_all_reduce=False):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        super().finish_grad_sync()

    def load_state_dict(self, state_dict, strict=True):
        """
        No-op because tensors are already loaded in-place by
        `_load_base_checkpoint` with FSDP2."""
        pass
