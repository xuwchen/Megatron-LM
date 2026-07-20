# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from examples.multimodal_dev.data.mock import MockQwen35VLDataset


def _build_mock_dataset(random_seed: int = 1234) -> MockQwen35VLDataset:
    return MockQwen35VLDataset(
        num_samples=16,
        seq_length=20,
        image_seq_length=2,
        vocab_size=1024,
        image_size=32,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        random_seed=random_seed,
    )


def _assert_samples_equal(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


def test_mock_multimodal_sample_is_deterministic_by_index():
    dataset = _build_mock_dataset()
    torch.manual_seed(999)
    initial_rng_state = torch.random.get_rng_state().clone()

    first = dataset[3]
    second = dataset[3]

    _assert_samples_equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), initial_rng_state)


def test_mock_multimodal_sample_changes_with_index_and_seed():
    dataset = _build_mock_dataset()
    different_seed_dataset = _build_mock_dataset(random_seed=1235)

    sample = dataset[3]
    different_index_sample = dataset[4]
    different_seed_sample = different_seed_dataset[3]

    assert not torch.equal(sample["input_ids"], different_index_sample["input_ids"])
    assert not torch.equal(sample["pixel_values"], different_index_sample["pixel_values"])
    assert not torch.equal(sample["input_ids"], different_seed_sample["input_ids"])
    assert not torch.equal(sample["pixel_values"], different_seed_sample["pixel_values"])
