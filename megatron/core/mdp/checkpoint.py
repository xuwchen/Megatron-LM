# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP checkpoint facade: synchronous global torch_dist weight-only save/load.

Logical keys: ``language_model.*`` stays with the decoder chunks (PP/VPP
shards, decoder DP-CP replica metadata, produced by the native checkpoint
path); ``vision_model.*`` comes from the encoder DDP with **encoder WORLD**
replica metadata — one logical copy replicated on every rank. Plans, leaves,
forward handles, autograd graphs, and communication handles are never
persisted; optimizer, LR-scheduler, and RNG state are excluded (weight-only
restart, not exact resume).
"""

from typing import Mapping

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.mdp.errors import MdpCheckpointError

#: The state-dict key the encoder state travels under. It sits next to the
#: native ``model``/``model<N>`` keys so the sharded save/load skeleton is
#: symmetric between save and load.
ENCODER_STATE_KEY = "mdp_vision_model"

#: Vision MLP weight/bias keys that zero_pad_vision_mlp_channels() (encoder.py)
#: may have zero-padded past their real (checkpoint) size. Suffix-matched
#: against sharded_state_dict keys, mirroring how
#: LanguageModule.sharded_state_dict() marks the padded vocab embedding/output
#: layer (megatron/core/models/common/language_module/language_module.py).
_VISION_FFN_PADDED_KEY_SUFFIXES = (
    ".linear_fc1.weight",
    ".linear_fc1.bias",
    ".linear_fc2.weight",
)


def _mark_vision_ffn_padding_shape_mismatch(state_dict: Mapping) -> None:
    """Let vision MLP fc1/fc2 tensors load from a smaller (official, unpadded)
    checkpoint into a --mdp-zero-pad-vision-ffn-padded model.

    Sets ``allow_shape_mismatch = True`` on every ``linear_fc1``/``linear_fc2``
    ShardedTensor, so MCore's torch_dist strategy zero-initializes the target
    buffer, then copies only the overlapping (real, unpadded) prefix from the
    checkpoint (see ``strategies/torch.py``'s ``_mcore_to_dcp_compatible_tensor``:
    "if allow_shape_mismatch is True, the data is initialized with zeros prior
    to loading"). Combined with zero_pad_vision_mlp_channels() already
    establishing zero-padding as a training-time invariant, this makes the two
    directions consistent: a checkpoint saved from a padded model round-trips
    exactly, and a checkpoint saved from the real (unpadded) official
    architecture loads cleanly into a padded model with the new channels
    zero-initialized -- exactly the invariant zero_pad_vision_mlp_channels()
    already establishes at construction time for training-from-scratch.
    Symmetric with LanguageModule.sharded_state_dict()'s vocab-padding
    handling, which uses the same mechanism.
    """
    marked = 0
    for key, value in state_dict.items():
        if isinstance(value, ShardedTensor) and key.endswith(_VISION_FFN_PADDED_KEY_SUFFIXES):
            value.allow_shape_mismatch = True
            marked += 1
    if marked == 0:
        raise MdpCheckpointError(
            "MDP: zero_pad_vision_ffn=True but no linear_fc1/linear_fc2 "
            "ShardedTensor found in the encoder state dict violates: at least "
            "one vision MLP layer to mark. Check that the encoder spec still "
            "names its FFN submodules linear_fc1/linear_fc2."
        )


def encoder_sharded_state_dict(encoder_ddp, *, vision_ffn_may_be_padded: bool = False) -> Mapping:
    """The encoder's sharded model-weight state with WORLD replica metadata.

    The encoder is fully replicated: its replica domain is WORLD, not the
    decoder's DP-CP group — reusing the decoder metadata here would make
    every PP stage claim a distinct (wrong) replica coordinate.

    ``vision_ffn_may_be_padded`` should be the run's
    ``MdpConfig.zero_pad_vision_ffn`` value: when true, the vision MLP's
    linear_fc1/linear_fc2 tensors are marked shape-mismatch-tolerant (see
    ``_mark_vision_ffn_padding_shape_mismatch``) so an official (unpadded)
    checkpoint loads into this padded model. Left false by default so an
    unrelated real shape mismatch (a genuine config error) still fails loudly
    instead of being silently zero-filled.
    """
    state_dict = encoder_ddp.sharded_state_dict(
        prefix="vision_model.",
        metadata={"dp_cp_group": torch.distributed.group.WORLD},
    )
    if vision_ffn_may_be_padded:
        _mark_vision_ffn_padding_shape_mismatch(state_dict)
    return state_dict


def add_encoder_state(
    state_dict: dict, encoder_ddp, *, vision_ffn_may_be_padded: bool = False
) -> dict:
    """Add the encoder weights to a torch_dist checkpoint state dict.

    See :func:`encoder_sharded_state_dict` for ``vision_ffn_may_be_padded``.
    """
    if ENCODER_STATE_KEY in state_dict:
        raise MdpCheckpointError(
            f"MDP: state dict already contains {ENCODER_STATE_KEY!r}; the encoder "
            "state must be contributed exactly once."
        )
    state_dict[ENCODER_STATE_KEY] = encoder_sharded_state_dict(
        encoder_ddp, vision_ffn_may_be_padded=vision_ffn_may_be_padded
    )
    return state_dict


def assert_weight_only_checkpoint(args) -> None:
    """Reject non-weight-only checkpoint configurations at startup.

    MDP persists model weights only: optimizer, LR-scheduler, and RNG state
    start fresh after load. The native flags express exactly that.
    """
    problems = []
    save_or_load = (
        getattr(args, "save", None) is not None or getattr(args, "load", None) is not None
    )
    if save_or_load:
        # Design doc section 12: only the synchronous, persistent, global
        # torch_dist mode is supported. Asynchronous, non-persistent, and
        # constant-structure caching modes are rejected at startup (scoped to
        # save/load so checkpoint-free runs are unaffected by defaults).
        if getattr(args, "async_save", False):
            problems.append("no --async-save (asynchronous save is unsupported)")
        if getattr(args, "non_persistent_ckpt_type", None) is not None:
            problems.append(
                "no --non-persistent-ckpt-type (non-persistent checkpoints are "
                "unsupported)"
            )
        if getattr(args, "ckpt_assume_constant_structure", False):
            problems.append(
                "no --ckpt-assume-constant-structure (MDP's plan-derived "
                "structures change per iteration; a cached structure goes stale)"
            )
    if getattr(args, "save", None) is not None:
        if not getattr(args, "no_save_optim", False):
            problems.append("--no-save-optim")
        if not getattr(args, "no_save_rng", False):
            problems.append("--no-save-rng")
        # Megatron defaults ckpt_fully_parallel_save=True; the fully-parallel
        # path shards across one DP-CP group for every child, which is wrong
        # for the encoder's WORLD replica domain. Scoped to save/load so runs
        # that never touch a checkpoint are not rejected by the default.
        if getattr(args, "ckpt_fully_parallel_save", False):
            problems.append("--no-ckpt-fully-parallel-save")
    if getattr(args, "load", None) is not None:
        if not getattr(args, "no_load_optim", False):
            problems.append("--no-load-optim")
        if not getattr(args, "no_load_rng", False):
            problems.append("--no-load-rng")
        if getattr(args, "ckpt_fully_parallel_load", False):
            problems.append("--no-ckpt-fully-parallel-load (or omit --ckpt-fully-parallel-load)")
    if problems:
        raise MdpCheckpointError(
            "MDP: the checkpoint facade is a weight-only restart contract; run with "
            + " ".join(problems)
            + ". Optimizer, LR-scheduler, and RNG state are not persisted and start "
            "fresh after load."
        )
