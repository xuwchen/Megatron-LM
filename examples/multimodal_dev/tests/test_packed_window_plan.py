# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the packed-window plan generation kernel."""

import pytest

from examples.multimodal_dev.data.packed_window_plan import PackedWindowPlanGenerator

_BUCKET_MERGED = [49, 98, 98, 154, 196, 294, 588, 980, 1110, 1369]
_BUCKET_RAW = [4 * v for v in _BUCKET_MERGED]
_BUCKET_WEIGHTS = [22, 9, 9, 14, 13, 13, 11, 7, 2, 1]


def _config(**overrides):
    config = {
        "doc_length": {
            "short": {"mean": 2048, "sigma": 1.2, "min": 64, "max": 32767},
            "long": {"mean": 131072, "sigma": 0.8, "min": 32768, "max": 524288},
            "long_component_text_token_share": 0.25,
        },
        "p_text": 0.41,
        "image_density": {"mean_per_text_token": 0.00075, "gamma_shape": 1.0},
    }
    config.update(overrides)
    return config


def _make(seq_length=4096, num_windows=64, seed=1234, **config_overrides):
    return PackedWindowPlanGenerator(
        seq_length=seq_length,
        num_windows=num_windows,
        seed=seed,
        config=_config(**config_overrides),
        bucket_merged_tokens=_BUCKET_MERGED,
        bucket_raw_patches=_BUCKET_RAW,
        bucket_weights=_BUCKET_WEIGHTS,
    )


def test_every_window_sums_to_seq_length_and_atoms_never_straddle():
    generator = _make(num_windows=256)
    for idx in range(len(generator)):
        plan = generator.window(idx)
        assert sum(length for _, length in plan.segments) == generator.seq_length
        previous_end = -1
        for atom in plan.atoms:
            assert atom.offset >= 0
            assert atom.offset + 1 + atom.merged_tokens <= generator.seq_length
            assert atom.offset > previous_end  # in order, non-overlapping
            previous_end = atom.offset + atom.merged_tokens
        assert 0 <= plan.fill_tokens <= generator.seq_length


def test_plan_is_deterministic():
    lhs, rhs = _make(num_windows=64), _make(num_windows=64)
    assert lhs.total_docs == rhs.total_docs
    assert lhs.total_fill_tokens == rhs.total_fill_tokens
    for idx in range(64):
        assert lhs.window(idx) == rhs.window(idx)


def test_spill_preserves_overtaken_atoms_and_order(monkeypatch):
    # Crafted doc: text 64, atom A (V=63 -> size 64) at offset 60, atom B
    # (same size) at nominal offset 62. With S=100, A does not fit before
    # the first window line: FILL-1 pulls the 4 remaining text tokens
    # forward (crossing B's nominal offset), FILL-2 pads 36 tokens, and A
    # lands at window 1 offset 0. B is overtaken but must survive in order:
    # it spills again (no text left) and lands at window 2 offset 0.
    crafted = {0: (64, (60, 62), (0, 0))}

    def fake_draw(self, doc_id):
        if doc_id in crafted:
            return crafted[doc_id]
        return 1000, (), ()  # plain text filler docs

    monkeypatch.setattr(PackedWindowPlanGenerator, "_draw_doc", fake_draw)
    generator = PackedWindowPlanGenerator(
        seq_length=100,
        num_windows=4,
        seed=7,
        config=_config(),
        bucket_merged_tokens=[63],
        bucket_raw_patches=[252],
        bucket_weights=[1],
    )

    atoms = [atom for idx in range(4) for atom in generator.window(idx).atoms]
    assert [(a.window, a.offset, a.doc_id, a.index_in_doc) for a in atoms] == [
        (1, 0, 0, 0),
        (2, 0, 0, 1),
    ]
    assert generator.total_spilled_atoms == 2
    # Window 0: 60 text + 4 pulled text (FILL-1) + 36 boundary_fill (FILL-2).
    assert generator.window(0).fill_tokens == 36
    # Window 2: atom B (64) then no doc-0 text remains -> next doc's text.
    assert sum(length for _, length in generator.window(2).segments) == 100


def test_config_validation():
    with pytest.raises(ValueError, match="exceeds the window size"):
        _make(seq_length=1024)  # largest atom 1370 > 1024
    with pytest.raises(ValueError, match="disjoint"):
        _make(
            doc_length={
                "short": {"mean": 2048, "sigma": 1.2, "min": 64, "max": 40000},
                "long": {"mean": 131072, "sigma": 0.8, "min": 32768, "max": 524288},
                "long_component_text_token_share": 0.25,
            }
        )
    with pytest.raises(ValueError, match="must be < 1"):
        _make(
            doc_length={
                "short": {"mean": 2048, "sigma": 1.2, "min": 64, "max": 32767},
                "long": {"mean": 131072, "sigma": 0.8, "min": 32768, "max": 524288},
                "long_component_text_token_share": 1.0,
            }
        )


def test_long_component_share_solves_p_long():
    assert (
        _make(
            doc_length={
                "short": {"mean": 2048, "sigma": 1.2, "min": 64, "max": 32767},
                "long": {"mean": 131072, "sigma": 0.8, "min": 32768, "max": 524288},
                "long_component_text_token_share": 0.0,
            }
        ).p_long
        == 0.0
    )
    generator = _make()  # share 0.25
    expected = (0.25 * 2048 / (0.75 * 131072)) / (1 + 0.25 * 2048 / (0.75 * 131072))
    assert abs(generator.p_long - expected) < 1e-12
    assert 0.004 < generator.p_long < 0.007  # reviewer's ~0.52% doc share


def test_fill_accounting_is_consistent():
    generator = _make(num_windows=512)
    assert generator.total_fill_tokens == sum(
        generator.window(idx).fill_tokens for idx in range(len(generator))
    )
    assert 0.0 <= generator.boundary_fill_fraction <= 0.005  # spec ceiling
    assert 0.0 <= generator.atom_spill_fraction < 1.0
