# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hand the recompute of a KDA layer the initial state its forward pass already merged.

With ``gdn`` in ``recompute_modules`` a KDA layer runs its computation twice per step: once
without gradients in the forward pass and once with gradients when the backward pass reaches
the layer. With chunk-wise context parallelism each run obtains the layer's initial state on
this rank from the other ranks of the CP group (FLA's
``chunk_gated_delta_rule_fwd_h_pre_process``: a summary kernel over the local tokens, an
all-gather of the summaries and a merge over the ranks crossed). The second run has the same
input, the same weights and the same quantization state as the first, so it computes the same
state again.

``ForwardStateSlot`` holds the state of one forward call of one layer between its two runs.
``run`` makes a slot the active one while the layer's computation runs, and the wrapper that
``install`` puts around FLA's exchange stores the state in the first run and returns it in
the second, which then skips the three steps. Every rank of a CP group takes the same branch
(it depends on the run's number and the gradient mode only), so the all-gather is skipped by
all of them or by none. Anything that is not the forward-then-recompute pair computes as
before.

The state is ``[N, HV, K, V]`` float32 per layer and is held from the forward pass to the
layer's recompute.

Set ``MEGATRON_KDA_FORWARD_STATE_REUSE_CHECK=1`` to make the second run compute the state as
well and raise if it is not bit-identical to the stored one (validation runs; it keeps the
work the reuse is there to save).
"""

import functools
import os
import threading

import torch

CHECK_ENV = "MEGATRON_KDA_FORWARD_STATE_REUSE_CHECK"
COUNTS = {"stored": 0, "reused": 0, "checked": 0, "computed": 0}
_LOCAL = threading.local()
_MARK = "_kda_forward_state_reuse"


class ForwardStateSlot:
    """The merged initial state of one forward call of one KDA layer, between its two runs."""

    __slots__ = ("state", "runs", "phase")

    def __init__(self):
        self.state = None
        self.runs = 0
        self.phase = None


def check_enabled() -> bool:
    """Whether the recompute verifies the stored state instead of trusting it."""
    return os.environ.get(CHECK_ENV, "0") == "1"


def run(slot, function, *args, **kwargs):
    """Call ``function`` with ``slot`` active; with ``slot=None`` just call it.

    The first run of a slot stores if it runs without gradients, as a checkpoint's forward
    does. The second takes if it runs with gradients, as the recompute does. Every other run
    computes, and drops whatever the slot held.
    """
    if slot is None:
        return function(*args, **kwargs)
    slot.runs += 1
    grad_enabled = torch.is_grad_enabled()
    if slot.runs == 1 and not grad_enabled:
        slot.phase = "store"
    elif slot.runs == 2 and grad_enabled:
        slot.phase = "take"
    else:
        slot.phase = "compute"
        slot.state = None
    previous = getattr(_LOCAL, "slot", None)
    _LOCAL.slot = slot
    try:
        return function(*args, **kwargs)
    finally:
        _LOCAL.slot = previous
        if slot.phase != "store":
            slot.state = None


def wrap(pre_process):
    """``pre_process`` (FLA's forward state exchange) with the reuse described above."""

    @functools.wraps(pre_process)
    def reusing(*args, **kwargs):
        slot = getattr(_LOCAL, "slot", None)
        context = kwargs.get("context")
        if slot is None or context is None or getattr(context, "group", None) is None:
            return pre_process(*args, **kwargs)
        if slot.phase == "store":
            if slot.state is not None:
                raise RuntimeError(
                    "The KDA forward state exchange ran twice in one run of a layer; "
                    "kda_recompute_reuse_forward_state expects one exchange per run."
                )
            slot.state = pre_process(*args, **kwargs)
            COUNTS["stored"] += 1
            return slot.state
        stored, slot.state = slot.state, None
        if slot.phase != "take" or stored is None:
            COUNTS["computed"] += 1
            return pre_process(*args, **kwargs)
        if check_enabled():
            fresh = pre_process(*args, **kwargs)
            COUNTS["checked"] += 1
            if fresh.shape != stored.shape or not torch.equal(fresh, stored):
                difference = (
                    float((fresh.double() - stored.double()).abs().max())
                    if fresh.shape == stored.shape
                    else float("inf")
                )
                raise RuntimeError(
                    "The KDA initial state of the recompute differs from the state of the "
                    f"forward pass (largest absolute difference {difference:.6e}, shapes "
                    f"{tuple(fresh.shape)} and {tuple(stored.shape)}); "
                    "kda_recompute_reuse_forward_state must not be used here."
                )
            return stored
        COUNTS["reused"] += 1
        return stored

    setattr(reusing, _MARK, True)
    return reusing


def install() -> bool:
    """Wrap the forward state exchange that FLA's KDA forward calls. Idempotent; returns
    whether this call installed the wrapper."""
    import fla.ops.kda.chunk_fwd as chunk_fwd

    current = chunk_fwd.chunk_gated_delta_rule_fwd_h_pre_process
    if getattr(current, _MARK, False):
        return False
    chunk_fwd.chunk_gated_delta_rule_fwd_h_pre_process = wrap(current)
    return True
