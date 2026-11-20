# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The KDA recompute takes the initial state its forward pass merged (no FLA, no GPU)."""

import sys
import types

import pytest
import torch

from megatron.core.ssm.gated_delta_net import forward_state_reuse as reuse
from megatron.core.transformer import TransformerConfig

CONTEXT = types.SimpleNamespace(group=object())


class Exchange:
    """Stands for FLA's forward state exchange: counts its calls, returns a new tensor each."""

    def __init__(self, drift=0.0):
        self.calls = 0
        self.drift = drift

    def __call__(self, k=None, context=None, **kwargs):
        self.calls += 1
        return torch.full((1, 2, 3, 3), 1.5) + self.drift * (self.calls - 1)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(reuse.CHECK_ENV, raising=False)
    for name in reuse.COUNTS:
        reuse.COUNTS[name] = 0
    yield
    assert getattr(reuse._LOCAL, "slot", None) is None


def layer(wrapped, context=CONTEXT):
    """One run of a layer's computation: the exchange once, as FLA's KDA forward calls it."""
    return lambda: wrapped(k=None, context=context)


def forward_then_recompute(wrapped, slot, context=CONTEXT):
    with torch.no_grad():
        first = reuse.run(slot, layer(wrapped, context))
    with torch.enable_grad():
        second = reuse.run(slot, layer(wrapped, context))
    return first, second


def test_the_option_is_off_by_default():
    assert (
        TransformerConfig.__dataclass_fields__["kda_recompute_reuse_forward_state"].default is False
    )


def test_the_recompute_takes_the_state_of_the_forward_pass_without_an_exchange():
    exchange = Exchange(drift=1.0)
    first, second = forward_then_recompute(reuse.wrap(exchange), reuse.ForwardStateSlot())
    assert exchange.calls == 1 and second is first
    assert reuse.COUNTS == {"stored": 1, "reused": 1, "checked": 0, "computed": 0}


def test_without_a_slot_both_runs_exchange():
    """``run(None, ...)`` is the layer with the option off."""
    exchange = Exchange()
    forward_then_recompute(reuse.wrap(exchange), None)
    assert exchange.calls == 2 and reuse.COUNTS["stored"] == 0


def test_nothing_is_kept_after_the_recompute_and_a_third_run_computes():
    exchange, slot = Exchange(), reuse.ForwardStateSlot()
    wrapped = reuse.wrap(exchange)
    forward_then_recompute(wrapped, slot)
    assert slot.state is None
    with torch.enable_grad():
        reuse.run(slot, layer(wrapped))
    assert exchange.calls == 2 and reuse.COUNTS["computed"] == 1


def test_each_forward_call_has_its_own_state():
    exchange = Exchange(drift=1.0)
    wrapped = reuse.wrap(exchange)
    slots = [reuse.ForwardStateSlot(), reuse.ForwardStateSlot()]
    with torch.no_grad():
        firsts = [reuse.run(slot, layer(wrapped)) for slot in slots]
    with torch.enable_grad():
        # The backward pass reaches the layers in the reverse order.
        seconds = [reuse.run(slot, layer(wrapped)) for slot in reversed(slots)]
    assert exchange.calls == 2
    assert seconds[0] is firsts[1] and seconds[1] is firsts[0]
    assert not torch.equal(firsts[0], firsts[1])


@pytest.mark.parametrize("context", [None, types.SimpleNamespace(group=None)])
def test_without_a_cp_group_the_exchange_runs_unchanged(context):
    exchange, slot = Exchange(), reuse.ForwardStateSlot()
    forward_then_recompute(reuse.wrap(exchange), slot, context)
    assert exchange.calls == 2 and slot.state is None and reuse.COUNTS["stored"] == 0


def test_a_first_run_with_gradients_stores_nothing():
    """Without a checkpoint the computation runs once, with gradients: nothing to reuse."""
    exchange, slot = Exchange(), reuse.ForwardStateSlot()
    wrapped = reuse.wrap(exchange)
    with torch.enable_grad():
        reuse.run(slot, layer(wrapped))
        reuse.run(slot, layer(wrapped))
    assert exchange.calls == 2 and slot.state is None and reuse.COUNTS["stored"] == 0


def test_a_second_run_without_gradients_does_not_take_the_state():
    exchange, slot = Exchange(), reuse.ForwardStateSlot()
    wrapped = reuse.wrap(exchange)
    with torch.no_grad():
        reuse.run(slot, layer(wrapped))
        reuse.run(slot, layer(wrapped))
    assert exchange.calls == 2 and slot.state is None and reuse.COUNTS["reused"] == 0


def test_two_exchanges_in_the_storing_run_are_refused():
    wrapped = reuse.wrap(Exchange())
    with torch.no_grad(), pytest.raises(RuntimeError, match="ran twice"):
        reuse.run(
            reuse.ForwardStateSlot(),
            lambda: (wrapped(k=None, context=CONTEXT), wrapped(k=None, context=CONTEXT)),
        )


def test_check_mode_computes_again_and_accepts_an_identical_state(monkeypatch):
    monkeypatch.setenv(reuse.CHECK_ENV, "1")
    exchange = Exchange()
    first, second = forward_then_recompute(reuse.wrap(exchange), reuse.ForwardStateSlot())
    assert exchange.calls == 2 and second is first
    assert reuse.COUNTS == {"stored": 1, "reused": 0, "checked": 1, "computed": 0}


def test_check_mode_raises_on_a_different_state(monkeypatch):
    monkeypatch.setenv(reuse.CHECK_ENV, "1")
    wrapped = reuse.wrap(Exchange(drift=0.25))
    with pytest.raises(RuntimeError, match="largest absolute difference 2.500000e-01"):
        forward_then_recompute(wrapped, reuse.ForwardStateSlot())


def test_a_failing_run_leaves_no_active_slot():
    def failing():
        raise KeyError("x")

    with torch.no_grad(), pytest.raises(KeyError):
        reuse.run(reuse.ForwardStateSlot(), failing)
    assert getattr(reuse._LOCAL, "slot", None) is None


def test_install_wraps_the_name_fla_kda_forward_calls_once(monkeypatch):
    exchange = Exchange()
    chunk_fwd = types.ModuleType("fla.ops.kda.chunk_fwd")
    chunk_fwd.chunk_gated_delta_rule_fwd_h_pre_process = exchange
    for name in ("fla", "fla.ops", "fla.ops.kda"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.kda.chunk_fwd", chunk_fwd)
    sys.modules["fla.ops.kda"].chunk_fwd = chunk_fwd
    assert reuse.install() is True and reuse.install() is False
    forward_then_recompute(
        chunk_fwd.chunk_gated_delta_rule_fwd_h_pre_process, reuse.ForwardStateSlot()
    )
    assert exchange.calls == 1
