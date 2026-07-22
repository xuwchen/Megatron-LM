# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the variable-length Qwen3.5-VL mock dataset."""

from types import SimpleNamespace

import pytest
import torch

import megatron.training
from examples.multimodal_dev.data import mock_varlen
from examples.multimodal_dev.data.mock_varlen import (
    MockQwen35VLVarlenDataset,
    PackedWindowQwen35VLDataset,
    train_valid_test_varlen_datasets_provider,
)

_IMAGE_TOKEN_ID = 97
_VIDEO_TOKEN_ID = 98
_VISION_START_TOKEN_ID = 96
_VOCAB_SIZE = 100
_IMAGE_SIZE = 8
_PATCH_SIZE = 2
_TEMPORAL_PATCH_SIZE = 2
_SPATIAL_MERGE_SIZE = 2
_GRID_THW = (1, 4, 4)
_NUM_PATCHES = 16
_NUM_MERGED_TOKENS = 4
_PIXEL_DIM = 24
_MIN_SEQ_LENGTH = 7


def _make_dataset(**overrides):
    kwargs = {
        "num_samples": 32,
        "seq_length": 32,
        "seed": 1234,
        "vocab_size": _VOCAB_SIZE,
        "image_token_id": _IMAGE_TOKEN_ID,
        "video_token_id": _VIDEO_TOKEN_ID,
        "vision_start_token_id": _VISION_START_TOKEN_ID,
        "image_size": _IMAGE_SIZE,
        "patch_size": _PATCH_SIZE,
        "temporal_patch_size": _TEMPORAL_PATCH_SIZE,
        "spatial_merge_size": _SPATIAL_MERGE_SIZE,
    }
    kwargs.update(overrides)
    return MockQwen35VLVarlenDataset(**kwargs)


def _file_config(path):
    return {"mode": "file", "path": str(path)}


def _bucket_config(*resolutions, weights=None):
    config = {"mode": "buckets", "resolutions": [list(size) for size in resolutions]}
    if weights is not None:
        config["weights"] = list(weights)
    return config


def _count_config(counts, weights):
    return {"mode": "categorical", "counts": list(counts), "weights": list(weights)}


def _modality_config(modalities, weights):
    return {"mode": "categorical", "modalities": list(modalities), "weights": list(weights)}


def _assert_samples_equal(lhs, rhs):
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        assert torch.equal(lhs[key], rhs[key]), key


def _assert_multi_image_contract(sample):
    input_ids = sample["input_ids"]
    grids = sample["image_grid_thw"]
    pixel_values = sample["pixel_values"]
    vision_starts = torch.where(input_ids == _VISION_START_TOKEN_ID)[0].tolist()

    assert tuple(grids.shape) == (len(vision_starts), 3)
    assert len(vision_starts) >= 1

    patch_offset = 0
    expected_image_tokens = 0
    block_ends = []
    for vision_start, (t, h, w) in zip(vision_starts, grids.tolist()):
        num_patches = t * h * w
        num_image_tokens = t * (h // _SPATIAL_MERGE_SIZE) * (w // _SPATIAL_MERGE_SIZE)
        image_start = vision_start + 1
        image_end = image_start + num_image_tokens

        assert torch.all(input_ids[image_start:image_end] == _IMAGE_TOKEN_ID)
        assert pixel_values[patch_offset : patch_offset + num_patches].shape == (
            num_patches,
            _PIXEL_DIM,
        )
        patch_offset += num_patches
        expected_image_tokens += num_image_tokens
        block_ends.append(image_end)

    assert patch_offset == pixel_values.shape[0]
    assert expected_image_tokens == int((input_ids == _IMAGE_TOKEN_ID).sum().item())
    assert int((input_ids == _VISION_START_TOKEN_ID).sum().item()) == grids.shape[0]
    assert all(end <= next_start for end, next_start in zip(block_ends, vision_starts[1:]))
    return vision_starts, block_ends


def test_provider_rejects_packed_hybridep_without_variable_token_padding(monkeypatch):
    args = SimpleNamespace(
        use_varlen_dataset=False,
        sequence_packing_scheduler=None,
        use_packed_sequence=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_pad_variable_tokens=False,
    )
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)

    with pytest.raises(ValueError, match="--moe-hybridep-pad-variable-tokens"):
        train_valid_test_varlen_datasets_provider((1, 1, 1))


def test_provider_forwards_image_count_and_placement_config(monkeypatch):
    args = SimpleNamespace(
        use_varlen_dataset=False,
        sequence_packing_scheduler=None,
        use_packed_sequence=False,
        use_vanilla_collate_fn=True,
        total_seq_length=32,
        seq_length=32,
        varlen_mock_dataset_config_json=None,
        mock_image_size_config_json=None,
        mock_image_count_config_json=('{"mode":"categorical","counts":[4],"weights":[1]}'),
        mock_image_placement="uniform",
        # The provider intentionally keeps the Qwen defaults for video and
        # vision-start IDs, so its vocabulary must contain those IDs.
        padded_vocab_size=248320,
        image_token_id=_IMAGE_TOKEN_ID,
        image_size=32,
        seed=2026,
    )
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)

    train_ds, val_ds, test_ds = train_valid_test_varlen_datasets_provider((1, 1, 1))

    for dataset in (train_ds, val_ds, test_ds):
        sample = dataset[0]
        assert sample["image_grid_thw"].shape == (4, 3)
        assert sample["image_grid_thw"].tolist() == [[1, 2, 2]] * 4
        assert sample["pixel_values"].shape == (16, 1536)
        assert int((sample["input_ids"] == dataset.vision_start_token_id).sum().item()) == 4
        assert int((sample["input_ids"] == dataset.image_token_id).sum().item()) == 4


def test_file_lengths_are_exact_and_repeat(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("11, 15\n19\n", encoding="utf-8")
    dataset = _make_dataset(num_samples=5, seq_length=20, length_config=_file_config(lengths_file))

    samples = [dataset[idx] for idx in range(len(dataset))]

    assert [sample["input_ids"].numel() for sample in samples] == [11, 15, 19, 11, 15]
    assert all(
        sample.keys() == {"input_ids", "labels", "loss_mask", "pixel_values", "image_grid_thw"}
        for sample in samples
    )
    assert all(sample["labels"].shape == sample["input_ids"].shape for sample in samples)
    assert all(sample["loss_mask"].shape == sample["input_ids"].shape for sample in samples)


@pytest.mark.parametrize("contents", ["11,,19\n", "\n", "11\n  \n19\n"])
def test_rejects_empty_csv_fields_and_rows(tmp_path, contents):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="must contain an integer in every field"):
        _make_dataset(length_config=_file_config(lengths_file))


def test_rejects_csv_header(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("length\n11\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain headerless integers"):
        _make_dataset(length_config=_file_config(lengths_file))


def test_image_geometry_matches_tokens_and_pixels(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("23\n", encoding="utf-8")
    dataset = _make_dataset(seq_length=23, length_config=_file_config(lengths_file))

    sample = dataset[0]
    input_ids = sample["input_ids"]
    grid_thw = sample["image_grid_thw"]
    pixel_values = sample["pixel_values"]
    t, h, w = grid_thw[0].tolist()

    assert tuple(grid_thw.shape) == (1, 3)
    assert (t, h, w) == _GRID_THW
    assert pixel_values.shape == (t * h * w, _PIXEL_DIM)
    assert pixel_values.shape == (_NUM_PATCHES, _PIXEL_DIM)

    expected_image_tokens = t * (h // _SPATIAL_MERGE_SIZE) * (w // _SPATIAL_MERGE_SIZE)
    assert expected_image_tokens == _NUM_MERGED_TOKENS
    assert int((input_ids == _IMAGE_TOKEN_ID).sum()) == expected_image_tokens
    assert int((input_ids == _VIDEO_TOKEN_ID).sum()) == 0

    vision_start = torch.where(input_ids == _VISION_START_TOKEN_ID)[0]
    assert vision_start.numel() == 1
    image_start = int(vision_start.item()) + 1
    assert torch.all(
        input_ids[image_start : image_start + expected_image_tokens] == _IMAGE_TOKEN_ID
    )
    assert input_ids.numel() == 23
    assert torch.all((0 <= input_ids) & (input_ids < _VOCAB_SIZE))


@pytest.mark.parametrize("num_images", [1, 2, 3, 4, 6, 8])
def test_one_hot_image_count_generates_requested_number(num_images):
    # The counts/weights-only shorthand is useful for forced smoke coverage.
    count_config = (
        {"counts": [num_images], "weights": [1]}
        if num_images == 4
        else _count_config([num_images], [1])
    )
    dataset = _make_dataset(num_samples=1, seq_length=64, image_count_config=count_config)

    sample = dataset[0]

    assert sample["image_grid_thw"].shape == (num_images, 3)
    assert int((sample["input_ids"] == _VISION_START_TOKEN_ID).sum().item()) == num_images
    _assert_multi_image_contract(sample)


def test_default_remains_one_fixed_centered_image(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("23\n", encoding="utf-8")
    dataset = _make_dataset(seq_length=23, length_config=_file_config(lengths_file))

    sample = dataset[0]
    vision_start = int(torch.where(sample["input_ids"] == _VISION_START_TOKEN_ID)[0].item())
    text_length = 23 - 1 - _NUM_MERGED_TOKENS

    assert sample["image_grid_thw"].tolist() == [list(_GRID_THW)]
    assert vision_start == text_length // 2
    _assert_multi_image_contract(sample)


def test_multi_image_dynamic_geometries_preserve_block_and_payload_order(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("96\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=2,
        seq_length=96,
        length_config=_file_config(lengths_file),
        image_count_config=_count_config([4], [1]),
        image_size_config=_bucket_config((8, 8), (8, 16), (16, 8), (16, 16)),
        image_placement="uniform",
        seed=2026,
    )

    sample = dataset[1]

    _assert_multi_image_contract(sample)
    geometries = dataset.image_geometry_sampler(1, num_images=4, max_total_merged_tokens=96 - 4 - 2)
    text_length = 96 - 4 - sum(geometry.num_merged_tokens for geometry in geometries)
    gaps = dataset._image_gaps(1, text_length=text_length, num_images=4)
    image_order = sorted(range(4), key=lambda image_idx: (gaps[image_idx], image_idx))
    expected_grids = [
        [1, geometries[image_idx].grid_h, geometries[image_idx].grid_w] for image_idx in image_order
    ]
    pixels_by_image = [
        torch.randn(
            geometry.total_patches,
            dataset.pixel_dim,
            generator=dataset._generator(1, stream=mock_varlen._PIXEL_VALUE_STREAM, item=image_idx),
        )
        for image_idx, geometry in enumerate(geometries)
    ]
    expected_pixels = torch.cat([pixels_by_image[image_idx] for image_idx in image_order])

    # Seed 2026 / index 1 mixes three distinct buckets with one repeat, then
    # uniform placement reorders their payloads into final token order.
    assert expected_grids == [[1, 4, 4], [1, 8, 4], [1, 4, 8], [1, 4, 4]]
    assert sample["image_grid_thw"].tolist() == expected_grids
    assert torch.equal(sample["pixel_values"], expected_pixels)


def test_uniform_placement_covers_all_text_gaps_and_allows_adjacent_images(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("22\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=1,
        seq_length=22,
        length_config=_file_config(lengths_file),
        image_count_config=_count_config([4], [1]),
        image_placement="uniform",
        seed=2026,
    )

    sample = dataset[0]
    vision_starts, block_ends = _assert_multi_image_contract(sample)

    # With two text tokens, seed 2026 / index 0 samples gaps [1, 0, 2, 0].
    # Stable gap sorting therefore covers start/middle/end and keeps the two
    # images at gap 0 adjacent.
    assert vision_starts == [0, 5, 11, 17]
    assert block_ends == [5, 10, 16, 22]


def test_text_only_modality_emits_empty_vision_payload(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("13\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=1,
        seq_length=13,
        length_config=_file_config(lengths_file),
        modality_config=_modality_config(["text_only"], [1]),
    )

    sample = dataset[0]
    input_ids = sample["input_ids"]

    assert input_ids.numel() == 13
    assert sample["pixel_values"].shape == (0, _PIXEL_DIM)
    assert sample["image_grid_thw"].shape == (0, 3)
    for token_id in (_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID):
        assert int((input_ids == token_id).sum().item()) == 0
    assert torch.equal(sample["labels"][:-1], input_ids[1:])
    assert sample["labels"][-1].item() == -100
    assert sample["loss_mask"].sum().item() == 12.0


def test_image_only_modality_has_no_text_tokens():
    dataset = _make_dataset(
        num_samples=1,
        seq_length=32,
        modality_config=_modality_config(["image_only"], [1]),
        image_count_config=_count_config([2], [1]),
    )

    sample = dataset[0]
    input_ids = sample["input_ids"]

    # 2 x (vision_start + 4 merged tokens); the sampled length is only a budget.
    assert input_ids.numel() == 2 * (1 + _NUM_MERGED_TOKENS)
    assert set(input_ids.tolist()) == {_IMAGE_TOKEN_ID, _VISION_START_TOKEN_ID}
    assert torch.all(sample["labels"] == -100)
    assert sample["loss_mask"].sum().item() == 0.0
    assert sample["pixel_values"].shape == (2 * _NUM_PATCHES, _PIXEL_DIM)
    _assert_multi_image_contract(sample)


def test_image_only_length_is_bounded_by_the_sampled_budget(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("23\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=1,
        seq_length=23,
        length_config=_file_config(lengths_file),
        modality_config=_modality_config(["image_only"], [1]),
        image_count_config=_count_config([1, 2, 3, 4], [1, 1, 1, 1]),
        image_size_config=_bucket_config((8, 8), (8, 16)),
    )

    sample = dataset[0]
    num_images = sample["image_grid_thw"].shape[0]
    merged_tokens = int((sample["input_ids"] == _IMAGE_TOKEN_ID).sum().item())

    assert sample["input_ids"].numel() == num_images + merged_tokens
    assert sample["input_ids"].numel() <= 23
    _assert_multi_image_contract(sample)


def test_short_length_renormalizes_over_feasible_modalities(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("2\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=1,
        seq_length=32,
        length_config=_file_config(lengths_file),
        modality_config=_modality_config(["interleaved", "text_only", "image_only"], [1, 1, 98]),
    )

    # Length 2 cannot host any image (1 + 4 merged tokens), so the sample must
    # renormalize onto text_only despite its low weight.
    sample = dataset[0]

    assert sample["input_ids"].numel() == 2
    assert sample["pixel_values"].shape == (0, _PIXEL_DIM)
    assert sample["image_grid_thw"].shape == (0, 3)


def test_modality_mix_is_deterministic_and_covers_all_modalities():
    kwargs = {
        "num_samples": 64,
        "seq_length": 48,
        "modality_config": _modality_config(
            ["interleaved", "text_only", "image_only"], [60, 30, 10]
        ),
        "seed": 2026,
    }
    dataset = _make_dataset(**kwargs)
    same_seed = _make_dataset(**kwargs)

    def classify(sample):
        has_image = sample["image_grid_thw"].shape[0] > 0
        special = {_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID}
        has_text = any(token not in special for token in sample["input_ids"].tolist())
        if has_image and has_text:
            return "interleaved"
        return "image_only" if has_image else "text_only"

    modalities = [classify(dataset[idx]) for idx in range(len(dataset))]

    assert set(modalities) == {"interleaved", "text_only", "image_only"}
    assert modalities == [classify(same_seed[idx]) for idx in range(len(same_seed))]
    for idx, modality in enumerate(modalities):
        sample = dataset[idx]
        if modality == "text_only":
            assert sample["pixel_values"].shape == (0, _PIXEL_DIM)
        elif modality == "image_only":
            assert sample["loss_mask"].sum().item() == 0.0
        else:
            assert sample["loss_mask"].sum().item() > 0.0
            _assert_multi_image_contract(sample)


@pytest.mark.parametrize(
    ("modality_config", "message"),
    [
        ({"mode": "uniform", "modalities": ["text_only"], "weights": [1]}, "categorical"),
        ({"mode": "categorical", "weights": [1]}, "modalities"),
        ({"mode": "categorical", "modalities": ["text_only"]}, "weights"),
        (_modality_config([], []), "non-empty"),
        (_modality_config(["video_only"], [1]), "Unsupported mock modality"),
        (_modality_config(["text_only", "text_only"], [1, 1]), "duplicate"),
        (_modality_config(["text_only", "interleaved"], [1]), "same length"),
        (_modality_config(["text_only"], [-1]), "non-negative"),
        (_modality_config(["text_only"], [0]), "positive"),
    ],
)
def test_rejects_invalid_modality_config(modality_config, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _make_dataset(modality_config=modality_config)


def test_rejects_non_dict_modality_config():
    with pytest.raises(TypeError, match="modality_config must be a dict"):
        _make_dataset(modality_config=["text_only"])


def test_provider_rejects_image_only_only_modality_profile(monkeypatch):
    args = SimpleNamespace(
        use_varlen_dataset=False,
        sequence_packing_scheduler=None,
        use_packed_sequence=False,
        use_vanilla_collate_fn=True,
        total_seq_length=32,
        seq_length=32,
        varlen_mock_dataset_config_json=None,
        mock_image_size_config_json=None,
        mock_image_count_config_json=None,
        mock_modality_config_json='{"modalities":["image_only"],"weights":[1]}',
        mock_image_placement="center",
        padded_vocab_size=248320,
        image_token_id=_IMAGE_TOKEN_ID,
        image_size=32,
        seed=2026,
    )
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)

    with pytest.raises(ValueError, match="text-bearing"):
        train_valid_test_varlen_datasets_provider((1, 1, 1))


def test_short_length_renormalizes_over_feasible_image_counts(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("7\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=1,
        seq_length=7,
        length_config=_file_config(lengths_file),
        image_count_config=_count_config([1, 4], [1, 99]),
        image_size_config=_bucket_config((8, 8), (8, 16)),
    )

    sample = dataset[0]

    assert sample["image_grid_thw"].tolist() == [[1, 4, 4]]
    _assert_multi_image_contract(sample)


def test_density_image_count_scales_with_sample_length(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("64\n" * 16 + "640\n" * 16, encoding="utf-8")
    dataset = _make_dataset(
        num_samples=32,
        seq_length=640,
        length_config=_file_config(lengths_file),
        image_count_config={"mode": "density", "images_per_1k_tokens": 20, "max_count": 64},
    )

    counts = [dataset[idx]["image_grid_thw"].shape[0] for idx in range(32)]
    short_counts, long_counts = counts[:16], counts[16:]

    # Poisson means: 20 * 64/1000 = 1.28 and 20 * 640/1000 = 12.8.
    assert all(1 <= count <= 5 for count in short_counts)
    assert all(5 <= count <= 25 for count in long_counts)
    short_mean = sum(short_counts) / len(short_counts)
    long_mean = sum(long_counts) / len(long_counts)
    assert 0.6 <= short_mean <= 2.2
    assert 9.0 <= long_mean <= 17.0
    for idx in (0, 16):
        _assert_multi_image_contract(dataset[idx])


def test_density_image_count_respects_max_count_and_budget(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("640\n", encoding="utf-8")
    # Poisson mean 12.8 clamps down to max_count.
    capped = _make_dataset(
        num_samples=4,
        seq_length=640,
        length_config=_file_config(lengths_file),
        image_count_config={"mode": "density", "images_per_1k_tokens": 20, "max_count": 3},
    )
    assert all(capped[idx]["image_grid_thw"].shape[0] == 3 for idx in range(4)), [
        capped[idx]["image_grid_thw"].shape[0] for idx in range(4)
    ]

    # Budget clamp: a 16x16 image costs 1 + 16 tokens, so a 64-token sample
    # fits at most (64 - 2) // 17 = 3 images regardless of the Poisson draw.
    lengths_file.write_text("64\n", encoding="utf-8")
    budget_bound = _make_dataset(
        num_samples=8,
        seq_length=64,
        length_config=_file_config(lengths_file),
        image_size=16,
        image_count_config={"mode": "density", "images_per_1k_tokens": 64, "max_count": 100},
    )
    counts = [budget_bound[idx]["image_grid_thw"].shape[0] for idx in range(8)]
    assert all(1 <= count <= 3 for count in counts), counts
    assert max(counts) == 3, counts
    assert budget_bound[0]["input_ids"].numel() == 64


def test_density_counts_are_deterministic():
    kwargs = {
        "num_samples": 8,
        "seq_length": 256,
        "image_count_config": {"mode": "density", "images_per_1k_tokens": 8, "max_count": 16},
        "seed": 2026,
    }
    dataset = _make_dataset(**kwargs)
    same_seed = _make_dataset(**kwargs)

    for idx in range(8):
        _assert_samples_equal(dataset[idx], same_seed[idx])


def test_geometric_density_counts_match_the_production_count_shape():
    sampler = mock_varlen._ImageCountSampler(
        {
            "mode": "density",
            "images_per_1k_tokens": 20,
            "max_count": 512,
            "distribution": "geometric",
        },
        seed=2026,
    )
    counts = [sampler(idx, sample_length=640, min_merged_tokens=0) for idx in range(1000)]

    # Geometric with mean 20 * 640 / 1000 = 12.8: mode-1 decreasing shape with
    # a long tail, unlike Poisson which concentrates around the mean.
    mean = sum(counts) / len(counts)
    assert 11.0 <= mean <= 15.0
    assert min(counts) == 1
    assert max(counts) >= 2 * 12.8
    below_mean = sum(1 for count in counts if count <= 12.8) / len(counts)
    assert 0.55 <= below_mean <= 0.72  # 1 - exp(-1) for geometric; ~0.5 for Poisson

    resampled = mock_varlen._ImageCountSampler(
        {
            "mode": "density",
            "images_per_1k_tokens": 20,
            "max_count": 512,
            "distribution": "geometric",
        },
        seed=2026,
    )
    assert counts == [resampled(idx, sample_length=640, min_merged_tokens=0) for idx in range(1000)]


def test_geometric_density_short_samples_stay_feasible(tmp_path):
    # Density-implied mean below one image (1 * 64 / 1000) must still draw a
    # valid count, and the feasibility clamp still applies on top.
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("64\n" * 8, encoding="utf-8")
    dataset = _make_dataset(
        num_samples=8,
        seq_length=64,
        length_config=_file_config(lengths_file),
        image_size=16,
        image_count_config={
            "mode": "density",
            "images_per_1k_tokens": 1,
            "max_count": 512,
            "distribution": "geometric",
        },
    )

    counts = [dataset[idx]["image_grid_thw"].shape[0] for idx in range(8)]
    assert all(1 <= count <= 3 for count in counts), counts
    _assert_multi_image_contract(dataset[0])


@pytest.mark.parametrize(
    ("image_count_config", "message"),
    [
        ({"mode": "density", "max_count": 8}, "images_per_1k_tokens"),
        ({"mode": "density", "images_per_1k_tokens": 1}, "max_count"),
        (
            {"mode": "density", "images_per_1k_tokens": 0, "max_count": 8},
            r"finite number in \(0, 64",
        ),
        (
            {"mode": "density", "images_per_1k_tokens": float("nan"), "max_count": 8},
            r"finite number in \(0, 64",
        ),
        (
            {"mode": "density", "images_per_1k_tokens": 100, "max_count": 8},
            r"finite number in \(0, 64",
        ),
        ({"mode": "density", "images_per_1k_tokens": 1, "max_count": 0}, r"integer in \[1, 1024\]"),
        (
            {"mode": "density", "images_per_1k_tokens": 1, "max_count": 2000},
            r"integer in \[1, 1024\]",
        ),
        (
            {"mode": "density", "images_per_1k_tokens": 1, "max_count": True},
            r"integer in \[1, 1024\]",
        ),
        (
            {
                "mode": "density",
                "images_per_1k_tokens": 1,
                "max_count": 8,
                "distribution": "binomial",
            },
            "'poisson' or 'geometric'",
        ),
    ],
)
def test_rejects_invalid_density_config(image_count_config, message):
    with pytest.raises(ValueError, match=message):
        _make_dataset(image_count_config=image_count_config)


def test_large_image_count_geometry_sampling_is_feasible_and_deterministic():
    # 6^48 ordered tuples overflow any integer counter; the float-normalized
    # completion counts must still sample a feasible tuple deterministically.
    config = _bucket_config((8, 8), (8, 16))
    sampler_kwargs = {"image_size": 8, "patch_size": 2, "spatial_merge_size": 2, "seed": 7}
    first = mock_varlen._ImageGeometrySampler(config, **sampler_kwargs)
    second = mock_varlen._ImageGeometrySampler(config, **sampler_kwargs)

    budget = 48 * 6  # binding: min sum 48*4=192, max sum 48*8=384 > 288
    geometries = first(3, num_images=48, max_total_merged_tokens=budget)

    assert len(geometries) == 48
    assert sum(geometry.num_merged_tokens for geometry in geometries) <= budget
    assert geometries == second(3, num_images=48, max_total_merged_tokens=budget)


def test_bucket_weights_shape_single_image_marginals():
    dataset = _make_dataset(
        num_samples=400,
        seq_length=32,
        image_count_config=_count_config([1], [1]),
        image_size_config=_bucket_config((8, 8), (8, 16), weights=[9, 1]),
    )

    small = sum(1 for idx in range(400) if dataset[idx]["image_grid_thw"].tolist() == [[1, 4, 4]])
    assert 0.84 <= small / 400 <= 0.96  # expected 0.9


def test_bucket_weights_apply_as_tuple_products():
    # Two buckets weighted 3:1 with an unconstrained budget: the first image
    # of each pair should be the small bucket with probability 3/4.
    dataset = _make_dataset(
        num_samples=1,
        seq_length=64,
        image_size_config=_bucket_config((8, 8), (8, 16), weights=[3, 1]),
    )
    sampler = dataset.image_geometry_sampler

    first_small = sum(
        1
        for idx in range(2000)
        if sampler(idx, num_images=2, max_total_merged_tokens=48)[0].num_merged_tokens == 4
    )
    assert 0.71 <= first_small / 2000 <= 0.79  # expected 0.75


def test_zero_weight_buckets_are_dropped():
    dataset = _make_dataset(
        num_samples=8,
        seq_length=32,
        image_count_config=_count_config([1, 2], [1, 1]),
        image_size_config=_bucket_config((8, 8), (8, 16), weights=[0, 1]),
    )

    assert dataset.image_geometry_sampler.min_merged_tokens == 8
    for idx in range(8):
        grids = dataset[idx]["image_grid_thw"].tolist()
        assert all(grid == [1, 4, 8] for grid in grids)


def test_uniform_bucket_weights_are_bit_identical_to_unweighted():
    kwargs = {
        "num_samples": 8,
        "seq_length": 48,
        "image_count_config": _count_config([1, 2, 3], [3, 2, 1]),
        "image_placement": "uniform",
        "seed": 2026,
    }
    unweighted = _make_dataset(image_size_config=_bucket_config((8, 8), (8, 16)), **kwargs)
    weighted = _make_dataset(
        image_size_config=_bucket_config((8, 8), (8, 16), weights=[2, 2]), **kwargs
    )

    for idx in range(8):
        _assert_samples_equal(unweighted[idx], weighted[idx])


def test_legacy_single_image_path_honors_explicit_weights():
    # No count config -> legacy path; explicit weights replace cyclic choice.
    dataset = _make_dataset(
        num_samples=400,
        seq_length=32,
        image_size_config=_bucket_config((8, 8), (8, 16), weights=[1, 9]),
    )

    large = sum(1 for idx in range(400) if dataset[idx]["image_grid_thw"].tolist() == [[1, 4, 8]])
    assert 0.84 <= large / 400 <= 0.96  # expected 0.9


@pytest.mark.parametrize(
    ("image_size_config", "message"),
    [
        (_bucket_config((8, 8), (8, 16), weights=[1]), "same length"),
        (_bucket_config((8, 8), weights=[-1]), "non-negative"),
        (_bucket_config((8, 8), (8, 16), weights=[0, 0]), "positive"),
    ],
)
def test_rejects_invalid_bucket_weights(image_size_config, message):
    with pytest.raises(ValueError, match=message):
        _make_dataset(image_size_config=image_size_config)


def test_absolute_vision_budget_bounds_every_sample():
    # Budget 9 fits exactly one 4-merged-token image (1 VS + 4 IMG = 5);
    # two images (10) exceed it, so counts must clamp to 1.
    dataset = _make_dataset(
        num_samples=16,
        seq_length=64,
        image_count_config={"mode": "density", "images_per_1k_tokens": 64, "max_count": 8},
        max_vision_tokens=9,
    )

    for idx in range(16):
        sample = dataset[idx]
        num_images = sample["image_grid_thw"].shape[0]
        vision_tokens = int((sample["input_ids"] == _IMAGE_TOKEN_ID).sum().item()) + num_images
        assert num_images == 1
        assert vision_tokens <= 9
    assert dataset.budget_stats.get("count_clamped_by_feasibility", 0) > 0


def test_fraction_vision_budget_scales_with_sample_length(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("20\n40\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=2,
        seq_length=40,
        length_config=_file_config(lengths_file),
        image_count_config=_count_config([1, 2, 3, 4], [1, 1, 1, 1]),
        max_vision_fraction=0.5,
    )

    for idx, length in enumerate((20, 40)):
        sample = dataset[idx]
        num_images = sample["image_grid_thw"].shape[0]
        vision_tokens = int((sample["input_ids"] == _IMAGE_TOKEN_ID).sum().item()) + num_images
        assert vision_tokens <= length // 2
        assert sample["input_ids"].numel() == length


def test_vision_budget_renormalizes_modality_to_text_only(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("20\n", encoding="utf-8")
    # fraction 0.1 -> budget 2 < smallest image (5): image-bearing infeasible.
    dataset = _make_dataset(
        num_samples=4,
        seq_length=64,
        length_config=_file_config(lengths_file),
        modality_config=_modality_config(["interleaved", "text_only"], [99, 1]),
        max_vision_fraction=0.1,
    )

    for idx in range(4):
        sample = dataset[idx]
        assert sample["image_grid_thw"].shape == (0, 3)
        assert sample["input_ids"].numel() == 20
    assert dataset.budget_stats.get("modality_dropped_by_vision_budget", 0) > 0


def test_vision_budget_constrains_geometry_tuples(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("64\n", encoding="utf-8")
    # Two images forced; budget 12 leaves 10 merged tokens -> only the
    # (4, 4) bucket pair fits (8 <= 10); any tuple with an 8-merged bucket
    # (12 > 10) must be filtered out.
    dataset = _make_dataset(
        num_samples=8,
        seq_length=64,
        length_config=_file_config(lengths_file),
        image_count_config=_count_config([2], [1]),
        image_size_config=_bucket_config((8, 8), (8, 16)),
        max_vision_tokens=12,
    )

    for idx in range(8):
        assert dataset[idx]["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 4]]


def test_loose_vision_budget_is_bit_identical_to_no_budget():
    kwargs = {
        "num_samples": 8,
        "seq_length": 48,
        "image_count_config": _count_config([1, 2, 3], [3, 2, 1]),
        "image_size_config": _bucket_config((8, 8), (8, 16)),
        "image_placement": "uniform",
        "seed": 2026,
    }
    unbudgeted = _make_dataset(**kwargs)
    budgeted = _make_dataset(max_vision_tokens=16384, max_vision_fraction=1.0, **kwargs)

    for idx in range(8):
        _assert_samples_equal(unbudgeted[idx], budgeted[idx])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_vision_tokens": 0}, "positive integer"),
        ({"max_vision_tokens": True}, "positive integer"),
        ({"max_vision_fraction": 0}, r"in \(0, 1\]"),
        ({"max_vision_fraction": 1.5}, r"in \(0, 1\]"),
        ({"max_vision_tokens": 4}, "cannot host the smallest"),
        ({"max_vision_fraction": 0.05, "seq_length": 32}, "cannot host the smallest"),
    ],
)
def test_rejects_invalid_vision_budget(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_dataset(**overrides)


def test_image_count_weights_are_normalized_internally():
    dataset = _make_dataset(image_count_config={"counts": [1, 2, 3, 4], "weights": [75, 15, 7, 3]})

    assert dataset.image_count_sampler.counts == (1, 2, 3, 4)
    assert dataset.image_count_sampler.weights == pytest.approx((0.75, 0.15, 0.07, 0.03))


def test_short_length_filters_joint_geometry_tuple(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("22\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=1,
        seq_length=22,
        length_config=_file_config(lengths_file),
        image_count_config=_count_config([4], [1]),
        image_size_config=_bucket_config((8, 8), (8, 16)),
    )

    sample = dataset[0]

    assert sample["image_grid_thw"].tolist() == [[1, 4, 4]] * 4
    assert sample["input_ids"].numel() == 22
    _assert_multi_image_contract(sample)


def test_labels_and_loss_mask_follow_shifted_targets(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("17\n", encoding="utf-8")
    dataset = _make_dataset(seq_length=17, length_config=_file_config(lengths_file))

    sample = dataset[0]
    input_ids = sample["input_ids"]
    raw_targets = torch.cat([input_ids[1:], torch.tensor([-100], dtype=torch.long)])
    ignored_targets = raw_targets == -100
    for token_id in (_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID):
        ignored_targets |= raw_targets == token_id

    expected_labels = raw_targets.masked_fill(ignored_targets, -100)
    expected_loss_mask = (~ignored_targets).to(torch.float32)
    assert torch.equal(sample["labels"], expected_labels)
    assert torch.equal(sample["loss_mask"], expected_loss_mask)

    vision_start_idx = int(torch.where(input_ids == _VISION_START_TOKEN_ID)[0].item())
    last_image_idx = int(torch.where(input_ids == _IMAGE_TOKEN_ID)[0][-1].item())
    assert sample["loss_mask"][vision_start_idx - 1].item() == 0.0
    assert sample["loss_mask"][vision_start_idx].item() == 0.0
    assert sample["loss_mask"][last_image_idx].item() == 1.0
    assert sample["labels"][-1].item() == -100
    assert sample["loss_mask"][-1].item() == 0.0


def test_samples_are_deterministic_and_access_order_independent():
    config = {
        "mode": "distribution",
        "type": "lognormal",
        "min_seq_len": 13,
        "max_seq_len": 32,
        "mean_seq_len": 22,
        "lognormal_sigma": 0.7,
    }
    image_size_config = _bucket_config((8, 8), (8, 16), (16, 8))
    dataset = _make_dataset(length_config=config, image_size_config=image_size_config, seed=2026)
    same_seed = _make_dataset(length_config=config, image_size_config=image_size_config, seed=2026)

    sample_7_before = dataset[7]
    _ = dataset[2]
    _ = dataset[19]
    sample_7_after = dataset[7]
    sample_7_fresh = same_seed[7]

    _assert_samples_equal(sample_7_before, sample_7_after)
    _assert_samples_equal(sample_7_before, sample_7_fresh)

    different_seed = _make_dataset(
        length_config=config, image_size_config=image_size_config, seed=2027
    )
    assert not torch.equal(sample_7_before["image_grid_thw"], different_seed[7]["image_grid_thw"])
    assert not torch.equal(sample_7_before["pixel_values"], different_seed[7]["pixel_values"])


def test_multi_image_random_streams_are_deterministic_and_distinct():
    kwargs = {
        "num_samples": 4,
        "seq_length": 64,
        "image_count_config": _count_config([4], [1]),
        "image_placement": "uniform",
        "seed": 2026,
    }
    dataset = _make_dataset(**kwargs)
    same_seed = _make_dataset(**kwargs)

    sample_0_before = dataset[0]
    _ = dataset[3]
    sample_0_after = dataset[0]
    sample_0_fresh = same_seed[0]
    sample_1 = dataset[1]

    _assert_samples_equal(sample_0_before, sample_0_after)
    _assert_samples_equal(sample_0_before, sample_0_fresh)

    patches_per_image = _NUM_PATCHES
    pixel_chunks = [
        sample_0_before["pixel_values"][offset : offset + patches_per_image]
        for offset in range(0, 4 * patches_per_image, patches_per_image)
    ]
    assert all(
        not torch.equal(lhs, rhs)
        for index, lhs in enumerate(pixel_chunks)
        for rhs in pixel_chunks[index + 1 :]
    )
    assert not torch.equal(pixel_chunks[0], sample_1["pixel_values"][:patches_per_image])


def test_dynamic_resolutions_vary_and_preserve_geometry(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("23\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=3,
        seq_length=23,
        length_config=_file_config(lengths_file),
        image_size_config=_bucket_config((8, 8), (8, 16), (16, 8)),
    )

    samples = [dataset[idx] for idx in range(len(dataset))]
    grids = {tuple(sample["image_grid_thw"][0].tolist()) for sample in samples}

    assert grids == {(1, 4, 4), (1, 4, 8), (1, 8, 4)}
    for sample in samples:
        t, h, w = sample["image_grid_thw"][0].tolist()
        expected_image_tokens = t * (h // _SPATIAL_MERGE_SIZE) * (w // _SPATIAL_MERGE_SIZE)
        assert sample["pixel_values"].shape == (t * h * w, _PIXEL_DIM)
        assert int((sample["input_ids"] == _IMAGE_TOKEN_ID).sum()) == expected_image_tokens


def test_short_lengths_only_select_feasible_resolutions(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("7\n11\n", encoding="utf-8")
    dataset = _make_dataset(
        num_samples=2,
        seq_length=11,
        length_config=_file_config(lengths_file),
        image_size_config=_bucket_config((8, 8), (8, 16)),
        seed=1234,
    )

    small, large = dataset[0], dataset[1]

    assert small["input_ids"].numel() == 7
    assert small["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert large["input_ids"].numel() == 11
    assert large["image_grid_thw"].tolist() == [[1, 4, 8]]


def test_truncated_lengths_have_no_endpoint_spikes_and_match_mean():
    # sigma=1.1 on a 2x window degenerated to ~57%/21% endpoint spikes under
    # the old clip semantics; truncated sampling must spread smoothly and
    # honor mean_seq_len as the post-truncation expectation.
    config = {
        "mode": "distribution",
        "type": "lognormal",
        "min_seq_len": 2048,
        "max_seq_len": 4096,
        "mean_seq_len": 3072,
        "lognormal_sigma": 1.1,
    }
    dataset = _make_dataset(num_samples=2000, seq_length=4096, length_config=config)

    lengths = [dataset.length_sampler(idx) for idx in range(2000)]

    at_min = sum(1 for length in lengths if length == 2048) / len(lengths)
    at_max = sum(1 for length in lengths if length == 4096) / len(lengths)
    assert at_min < 0.01
    assert at_max < 0.01
    mean = sum(lengths) / len(lengths)
    assert abs(mean - 3072) / 3072 < 0.02
    assert all(2048 <= length <= 4096 for length in lengths)


def test_truncated_lengths_survive_wide_windows():
    # Regression: on wide windows the mu bisection used to wander into a
    # region where the shifted CDF difference underflows to zero while the
    # window mass is still positive, collapsing every sampled length onto
    # max_seq_len (observed with [1024, 32768] mean 8192).
    config = {
        "mode": "distribution",
        "type": "lognormal",
        "min_seq_len": 1024,
        "max_seq_len": 32768,
        "mean_seq_len": 8192,
        "lognormal_sigma": 1.1,
    }
    dataset = _make_dataset(num_samples=2000, seq_length=32768, length_config=config)

    lengths = [dataset.length_sampler(idx) for idx in range(2000)]

    at_max = sum(1 for length in lengths if length == 32768) / len(lengths)
    assert at_max < 0.01
    mean = sum(lengths) / len(lengths)
    assert abs(mean - 8192) / 8192 < 0.05
    assert all(1024 <= length <= 32768 for length in lengths)


def test_degenerate_length_profiles_stay_constant():
    config = {
        "mode": "distribution",
        "type": "lognormal",
        "min_seq_len": 23,
        "max_seq_len": 23,
        "mean_seq_len": 23,
        "lognormal_sigma": 0.0,
    }
    dataset = _make_dataset(num_samples=4, seq_length=32, length_config=config)

    assert [dataset.length_sampler(idx) for idx in range(4)] == [23, 23, 23, 23]


def test_rejects_mean_on_window_bound_for_truncated_profile():
    config = {
        "mode": "distribution",
        "type": "lognormal",
        "min_seq_len": 16,
        "max_seq_len": 32,
        "mean_seq_len": 32,
        "lognormal_sigma": 1.1,
    }
    with pytest.raises(ValueError, match="strictly inside"):
        _make_dataset(length_config=config)


def test_lognormal_lengths_stay_in_bounds_and_vary():
    config = {
        "mode": "distribution",
        "type": "lognormal",
        "min_seq_len": 13,
        "max_seq_len": 40,
        "mean_seq_len": 24,
        "lognormal_sigma": 0.8,
    }
    dataset = _make_dataset(num_samples=128, seq_length=48, length_config=config)

    lengths = [dataset[idx]["input_ids"].numel() for idx in range(len(dataset))]

    assert min(lengths) >= 13
    assert max(lengths) <= 40
    assert len(set(lengths)) > 1


def test_sequence_length_larger_than_vocab_stays_in_vocab(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text("257\n", encoding="utf-8")
    dataset = _make_dataset(num_samples=1, seq_length=257, length_config=_file_config(lengths_file))

    input_ids = dataset[0]["input_ids"]

    assert input_ids.numel() == 257
    assert torch.all((0 <= input_ids) & (input_ids < _VOCAB_SIZE))


def test_rejects_sequence_length_that_cannot_hold_one_image():
    with pytest.raises(ValueError, match="too small for the smallest mock image"):
        _make_dataset(seq_length=_MIN_SEQ_LENGTH - 1)


def test_rejects_too_short_length_from_file(tmp_path):
    lengths_file = tmp_path / "lengths.csv"
    lengths_file.write_text(f"{_MIN_SEQ_LENGTH - 1}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must be in \[7, 32\]"):
        _make_dataset(length_config=_file_config(lengths_file))


def test_dynamic_config_does_not_validate_unused_fixed_image_size():
    dataset = _make_dataset(image_size=0, image_size_config=_bucket_config((8, 8)))

    assert dataset[0]["image_grid_thw"].tolist() == [[1, 4, 4]]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_size": 0}, "image_size must be positive"),
        ({"image_size": 10, "patch_size": 4}, "must be divisible by patch_size"),
        (
            {"image_size": 12, "patch_size": 2, "spatial_merge_size": 4},
            r"divisible by patch_size \* spatial_merge_size",
        ),
        ({"patch_size": 0}, "must be positive"),
    ],
)
def test_rejects_invalid_image_geometry(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_dataset(**overrides)


@pytest.mark.parametrize(
    ("image_size_config", "message"),
    [
        ({"mode": "buckets", "resolutions": []}, "non-empty list"),
        ({"mode": "buckets", "resolutions": [[8]]}, r"\[height, width\] pair"),
        ({"mode": "buckets", "resolutions": [[8, "16"]]}, "must be integers"),
        ({"mode": "buckets", "resolutions": [[0, 8]]}, "must be positive"),
        (
            {"mode": "buckets", "resolutions": [[8, 10]]},
            r"divisible by patch_size \* spatial_merge_size",
        ),
        ({"mode": "uniform", "resolutions": [[8, 8]]}, "expected 'buckets'"),
        ({"mode": "buckets"}, "requires a 'resolutions' field"),
    ],
)
def test_rejects_invalid_dynamic_resolution_config(image_size_config, message):
    with pytest.raises(ValueError, match=message):
        _make_dataset(image_size_config=image_size_config)


def test_rejects_non_dict_dynamic_resolution_config():
    with pytest.raises(TypeError, match="image_size_config must be a dict"):
        _make_dataset(image_size_config=[[8, 8]])


@pytest.mark.parametrize(
    ("image_count_config", "message"),
    [
        ({"mode": "uniform", "counts": [1], "weights": [1]}, "categorical"),
        ({"mode": "categorical", "weights": [1]}, "counts"),
        ({"mode": "categorical", "counts": [1]}, "weights"),
        (_count_config([], []), "non-empty"),
        (_count_config([0], [1]), "modality profile"),
        (_count_config([-1], [1]), r"must be in \[1, 8\]"),
        (_count_config([9], [1]), r"must be in \[1, 8\]"),
        (_count_config([1, 1], [1, 1]), "duplicate"),
        (_count_config([1, 2], [1]), "same length"),
        (_count_config([1], [-1]), "non-negative"),
        (_count_config([1], [float("nan")]), "finite"),
        (_count_config([1, 2], [0, 0]), "positive"),
    ],
)
def test_rejects_invalid_image_count_config(image_count_config, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _make_dataset(image_count_config=image_count_config)


def test_rejects_non_dict_image_count_config():
    with pytest.raises(TypeError, match="image_count_config must be a dict"):
        _make_dataset(image_count_config=[1, 2, 3, 4])


def test_rejects_invalid_image_placement():
    with pytest.raises(ValueError, match="center.*uniform"):
        _make_dataset(image_placement="random")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_token_id": _VOCAB_SIZE}, "must be in"),
        ({"image_token_id": 0}, "reserved for multimodal packing padding"),
        ({"video_token_id": 0}, "reserved for multimodal packing padding"),
        ({"vision_start_token_id": 0}, "reserved for multimodal packing padding"),
        ({"image_token_id": _VIDEO_TOKEN_ID}, "must be distinct"),
        (
            {"vocab_size": 4, "image_token_id": 1, "video_token_id": 2, "vision_start_token_id": 3},
            "usable non-special text token",
        ),
    ],
)
def test_rejects_invalid_vocabulary(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_dataset(**overrides)


# ===================================================================
# packed_window mode (v2): fixed-S windows over a mock document stream
# ===================================================================

_WINDOW_CONFIG = {
    "doc_length": {
        "short": {"mean": 24, "sigma": 0.8, "min": 8, "max": 63},
        "long": {"mean": 128, "sigma": 0.5, "min": 64, "max": 256},
        "long_component_text_token_share": 0.3,
    },
    "p_text": 0.4,
    "image_density": {"mean_per_text_token": 0.05, "gamma_shape": 1.0},
}


def _make_packed_window_dataset(**overrides):
    kwargs = {
        "num_samples": 32,
        "seq_length": 64,
        "seed": 1234,
        "vocab_size": _VOCAB_SIZE,
        "image_token_id": _IMAGE_TOKEN_ID,
        "video_token_id": _VIDEO_TOKEN_ID,
        "vision_start_token_id": _VISION_START_TOKEN_ID,
        "window_config": _WINDOW_CONFIG,
        "image_size_config": _bucket_config((8, 8), (8, 16)),
        "patch_size": _PATCH_SIZE,
        "temporal_patch_size": _TEMPORAL_PATCH_SIZE,
        "spatial_merge_size": _SPATIAL_MERGE_SIZE,
    }
    kwargs.update(overrides)
    return PackedWindowQwen35VLDataset(**kwargs)


class TestPackedWindowDataset:
    def test_windows_are_exactly_seq_length_with_matching_seq_lens(self):
        dataset = _make_packed_window_dataset()
        saw_atoms = saw_multi_segment = False
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert sample.keys() == {
                "input_ids",
                "labels",
                "loss_mask",
                "pixel_values",
                "image_grid_thw",
                "seq_lens",
            }
            assert sample["input_ids"].shape == (64,)
            assert int(sample["seq_lens"].sum().item()) == 64
            assert (sample["seq_lens"] > 0).all()
            if sample["image_grid_thw"].shape[0]:
                saw_atoms = True
                _assert_multi_image_contract(sample)
            if sample["seq_lens"].numel() > 1:
                saw_multi_segment = True
        assert saw_atoms and saw_multi_segment

    def test_segment_final_positions_have_no_targets(self):
        dataset = _make_packed_window_dataset()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            boundary = 0
            for segment_length in sample["seq_lens"].tolist():
                boundary += segment_length
                assert sample["labels"][boundary - 1].item() == -100
                assert sample["loss_mask"][boundary - 1].item() == 0.0

    def test_windows_are_deterministic(self):
        lhs = _make_packed_window_dataset()
        rhs = _make_packed_window_dataset()
        for idx in range(8):
            _assert_samples_equal(lhs[idx], rhs[idx])

    def test_requires_bucket_image_size_config(self):
        with pytest.raises(ValueError, match="buckets"):
            _make_packed_window_dataset(image_size_config=None)


def _packed_window_args(**overrides):
    args = SimpleNamespace(
        use_varlen_dataset=False,
        sequence_packing_scheduler=None,
        use_packed_sequence=True,
        use_vanilla_collate_fn=True,
        micro_batch_size=1,
        total_seq_length=64,
        seq_length=64,
        varlen_mock_dataset_config_json=(
            '{"mode":"packed_window",'
            '"doc_length":{"short":{"mean":24,"sigma":0.8,"min":8,"max":63},'
            '"long":{"mean":128,"sigma":0.5,"min":64,"max":256},'
            '"long_component_text_token_share":0.3},'
            '"p_text":0.4,'
            '"image_density":{"mean_per_text_token":0.05,"gamma_shape":1.0}}'
        ),
        mock_image_size_config_json='{"mode":"buckets","resolutions":[[32,32],[64,32]]}',
        mock_image_count_config_json=None,
        mock_modality_config_json=None,
        mock_max_vision_tokens=None,
        mock_max_vision_fraction=None,
        padded_vocab_size=248320,
        image_token_id=248056,
        seed=2026,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestPackedWindowProvider:
    def test_dispatches_to_packed_window_datasets(self, monkeypatch):
        monkeypatch.setattr(megatron.training, "get_args", lambda: _packed_window_args())
        train_ds, val_ds, test_ds = train_valid_test_varlen_datasets_provider((2, 1, 1))
        for dataset in (train_ds, val_ds, test_ds):
            assert isinstance(dataset, PackedWindowQwen35VLDataset)
        sample = train_ds[0]
        assert sample["input_ids"].shape == (64,)
        assert int(sample["seq_lens"].sum().item()) == 64
        # Distinct split seeds must not collapse the splits onto one stream.
        assert train_ds.seed != val_ds.seed != test_ds.seed

    def test_rejects_micro_batch_size_above_one(self, monkeypatch):
        monkeypatch.setattr(
            megatron.training, "get_args", lambda: _packed_window_args(micro_batch_size=2)
        )
        with pytest.raises(ValueError, match="micro_batch_size == 1"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    @pytest.mark.parametrize(
        "flag",
        [
            "mock_image_count_config_json",
            "mock_modality_config_json",
            "mock_max_vision_tokens",
            "mock_max_vision_fraction",
        ],
    )
    def test_rejects_obsolete_window_level_knobs(self, monkeypatch, flag):
        value = 16 if "tokens" in flag else 0.5 if "fraction" in flag else '{"mode":"density"}'
        monkeypatch.setattr(
            megatron.training, "get_args", lambda: _packed_window_args(**{flag: value})
        )
        with pytest.raises(ValueError, match="not supported in packed_window mode"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    def test_requires_explicit_long_component_share(self, monkeypatch):
        config = (
            '{"mode":"packed_window",'
            '"doc_length":{"short":{"mean":24,"sigma":0.8,"min":8,"max":63},'
            '"long":{"mean":128,"sigma":0.5,"min":64,"max":256}},'
            '"p_text":0.4,'
            '"image_density":{"mean_per_text_token":0.05,"gamma_shape":1.0}}'
        )
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _packed_window_args(varlen_mock_dataset_config_json=config),
        )
        with pytest.raises(ValueError, match="long_component_text_token_share"):
            train_valid_test_varlen_datasets_provider((1, 1, 1))
