# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the variable-length Qwen3.5-VL mock dataset."""

from types import SimpleNamespace

import pytest
import torch

import megatron.training
from examples.multimodal_dev.data.mock_varlen import (
    MockQwen35VLVarlenDataset,
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


def _bucket_config(*resolutions):
    return {"mode": "buckets", "resolutions": [list(size) for size in resolutions]}


def _assert_samples_equal(lhs, rhs):
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        assert torch.equal(lhs[key], rhs[key]), key


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
    ("overrides", "message"),
    [
        ({"image_token_id": _VOCAB_SIZE}, "must be in"),
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
