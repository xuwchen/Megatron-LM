# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the packed-window plan generation kernel."""

import pytest

from examples.multimodal_dev.data.packed_window_plan import (
    PARITY_PROFILE,
    PackedWindowPlanGenerator,
    context_scaled_doc_length,
)

_BUCKET_MERGED = [49, 98, 98, 154, 196, 294, 588, 980, 1110, 1369]
_BUCKET_RAW = [4 * v for v in _BUCKET_MERGED]
_BUCKET_WEIGHTS = [33, 9, 9, 14, 13, 13, 11, 7, 2, 1]

_COMPONENTS = [
    {"name": "short", "weight": 95, "min": 64, "max": 32767, "mean": 2048, "sigma": 1.2},
    {"name": "long", "weight": 5, "min": 32768, "max": 524288, "mean": 131072, "sigma": 0.8},
]


def _config(**overrides):
    config = {
        "doc_length": {"components": [dict(component) for component in _COMPONENTS]},
        "text_only_document_probability": 0.55,
        "image_poisson_rate_per_1k_text_tokens": 1.12,
        "image_density_gamma_shape": 0.38,
        # Small synthetic test configs trip the production ceiling; the
        # enforcement itself is covered by its dedicated test below.
        "max_boundary_fill_fraction": None,
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


def test_document_count_weights_are_document_proportions():
    # weights are per-document proportions (what corpus descriptions state):
    # [95, 5] must yield ~5% long documents by count, with no token-share
    # conversion involved.
    generator = _make(num_windows=2000)
    lengths = generator.doc_text_lengths
    long_fraction = sum(1 for length in lengths if length >= 32768) / len(lengths)
    assert 0.03 <= long_fraction <= 0.07
    assert generator.component_names == ("short", "long")


def test_sigma_zero_component_is_constant_length():
    generator = _make(
        num_windows=8,
        doc_length={
            "components": [
                {"name": "fixed", "weight": 1, "min": 64, "max": 128, "mean": 96, "sigma": 0}
            ]
        },
    )
    assert set(generator.doc_text_lengths) == {96}


def test_overlapping_components_are_legal():
    # Count-weight mixtures need no exclusivity semantics; two components
    # may cover the same range with different shapes.
    generator = _make(
        num_windows=8,
        doc_length={
            "components": [
                {"name": "a", "weight": 1, "min": 64, "max": 4096, "mean": 512, "sigma": 1.0},
                {"name": "b", "weight": 1, "min": 64, "max": 4096, "mean": 2048, "sigma": 0.5},
            ]
        },
    )
    assert len(generator) == 8


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
        config=_config(),  # ceiling already disabled in the test default
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


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        ({"doc_length": {"components": []}}, "non-empty list"),
        ({"doc_length": {}}, "non-empty list"),
        (
            {
                "doc_length": {
                    "components": [
                        {"name": "a", "weight": 1, "min": 64, "max": 128, "mean": 96, "sigma": 1},
                        {"name": "a", "weight": 1, "min": 64, "max": 128, "mean": 96, "sigma": 1},
                    ]
                }
            },
            "Duplicate doc_length component name",
        ),
        (
            {
                "doc_length": {
                    "components": [
                        {"name": "a", "weight": -1, "min": 64, "max": 128, "mean": 96, "sigma": 1}
                    ]
                }
            },
            "a.weight",
        ),
        (
            {
                "doc_length": {
                    "components": [
                        {"name": "a", "weight": 0, "min": 64, "max": 128, "mean": 96, "sigma": 1}
                    ]
                }
            },
            "positive sum",
        ),
        (
            {
                "doc_length": {
                    "components": [
                        {"name": "a", "weight": 1, "min": 128, "max": 64, "mean": 96, "sigma": 1}
                    ]
                }
            },
            "min <= max",
        ),
        (
            {
                "doc_length": {
                    "components": [
                        {"name": "a", "weight": 1, "min": 64, "max": 128, "mean": 256, "sigma": 1}
                    ]
                }
            },
            "must lie in",
        ),
        (
            {
                "doc_length": {
                    "components": [
                        {"name": "a", "weight": 1, "min": 64, "max": 128, "mean": 96, "sigma": -1}
                    ]
                }
            },
            "sigma",
        ),
        ({"text_only_document_probability": 1.5}, "must be in"),
        ({"image_poisson_rate_per_1k_text_tokens": 0}, "must be positive"),
        ({"image_density_gamma_shape": -1}, "must be positive"),
    ],
)
def test_config_validation(config_overrides, message):
    with pytest.raises(ValueError, match=message):
        _make(**config_overrides)


def test_oversized_atom_is_rejected():
    with pytest.raises(ValueError, match="exceeds the window size"):
        _make(seq_length=1024)  # largest atom 1370 > 1024


def test_interleaved_documents_always_carry_at_least_one_image():
    # Zero-truncated Poisson: text_only_document_probability is the EXACT
    # text-only probability, so with p_text=0 every document bears >=1 image.
    generator = _make(num_windows=64, text_only_document_probability=0.0)
    assert generator.total_atoms >= generator.total_docs


def test_boundary_fill_ceiling_is_enforced():
    # Atoms nearly as large as the window force heavy FILL-2 padding; the
    # ceiling turns that construction distortion into a loud failure.
    with pytest.raises(RuntimeError, match="acceptance ceiling"):
        PackedWindowPlanGenerator(
            seq_length=4096,
            num_windows=32,
            seed=1,
            config=_config(
                doc_length={
                    "components": [
                        {"name": "a", "weight": 1, "min": 64, "max": 128, "mean": 96, "sigma": 0.5}
                    ]
                },
                image_poisson_rate_per_1k_text_tokens=64,
                image_density_gamma_shape=1.0,
                max_boundary_fill_fraction=0.005,
            ),
            bucket_merged_tokens=[1369],
            bucket_raw_patches=[5476],
            bucket_weights=[1],
        )


def test_fill_accounting_is_consistent():
    generator = _make(num_windows=512)
    assert generator.total_fill_tokens == sum(
        generator.window(idx).fill_tokens for idx in range(len(generator))
    )
    assert 0.0 <= generator.boundary_fill_fraction <= 0.005  # spec ceiling
    assert 0.0 <= generator.atom_spill_fraction < 1.0


def test_component_unknown_fields_are_rejected():
    # A typoed field (e.g. "means") must not be silently ignored: the
    # benchmark would run a distribution the user did not write.
    components = [dict(component) for component in _COMPONENTS]
    components[0]["means"] = 128
    with pytest.raises(ValueError, match="unknown field"):
        _make(doc_length={"components": components})


def test_zero_weight_bucket_does_not_gate_feasibility():
    # An oversized bucket with weight 0 is disabled: it must not block a
    # window size that every drawable bucket fits.
    generator = PackedWindowPlanGenerator(
        seq_length=64,
        num_windows=8,
        seed=1234,
        config=_config(),
        bucket_merged_tokens=[100, 1],
        bucket_raw_patches=[400, 4],
        bucket_weights=[0, 1],
    )
    for idx in range(len(generator)):
        for atom in generator.window(idx).atoms:
            assert atom.merged_tokens == 1
    # A drawable oversized bucket still fails.
    with pytest.raises(ValueError, match="Largest image atom"):
        PackedWindowPlanGenerator(
            seq_length=64,
            num_windows=8,
            seed=1234,
            config=_config(),
            bucket_merged_tokens=[100, 1],
            bucket_raw_patches=[400, 4],
            bucket_weights=[1, 1],
        )


def test_component_min_max_must_be_strict_integers():
    for bad in ("128", 128.0, True, None):
        components = [
            {"name": "short", "weight": 1, "min": bad, "max": 256, "mean": 128, "sigma": 0.5}
        ]
        with pytest.raises(ValueError, match="must be an integer"):
            _make(doc_length={"components": components})
        components = [
            {"name": "short", "weight": 1, "min": 8, "max": bad, "mean": 128, "sigma": 0.5}
        ]
        with pytest.raises(ValueError, match="must be an integer"):
            _make(doc_length={"components": components})


class TestContextScaledPolicy:
    """The opt-in coverage policy: anchors are byte-identical to the
    frozen profiles; the curve is monotone, clamped, and bounded."""

    def test_4k_expansion_equals_the_parity_components(self):
        # PARITY_PROFILE itself is literal-locked in test_mock_varlen_dataset.
        assert (
            context_scaled_doc_length(4096)["components"]
            == PARITY_PROFILE["doc_length"]["components"]
        )

    def test_128k_expansion_equals_the_qualified_recipe_components(self):
        assert context_scaled_doc_length(131072)["components"] == [
            {
                "name": "short",
                "weight": 98.462,
                "min": 64,
                "max": 32767,
                "mean": 2048,
                "sigma": 1.2,
            },
            {
                "name": "long",
                "weight": 1.538,
                "min": 32768,
                "max": 524288,
                "mean": 131072,
                "sigma": 0.8,
            },
        ]

    def test_share_curve_is_monotone_and_clamped(self):
        weights = [
            context_scaled_doc_length(s)["components"][1]["weight"]
            for s in (1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)
        ]
        assert weights == sorted(weights)
        assert weights[0] == weights[1] == 0.518  # clamped below 4K
        assert weights[-1] == context_scaled_doc_length(524288)["components"][1]["weight"]

    def test_long_component_geometry_never_moves(self):
        for s in (4096, 32768, 131072, 524288):
            long_comp = context_scaled_doc_length(s)["components"][1]
            assert (long_comp["min"], long_comp["max"], long_comp["mean"], long_comp["sigma"]) == (
                32768,
                524288,
                131072,
                0.8,
            )

    def test_beyond_512k_requires_explicit_components(self):
        with pytest.raises(ValueError, match="explicit doc_length components"):
            context_scaled_doc_length(524289)
