# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Variable-length mock image-text data for Qwen3.5-VL training.

The generic :class:`MockVarlenDataset` introduced for text-only training
cannot transport ragged vision payloads through the core packing scheduler.
This provider therefore mirrors the raw per-sample contract of
``CordV2VLMDataset`` and leaves multimodal packing to
``multimodal_dev.forward_step.pack_or_pad_batch``.
"""

import csv
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from megatron.training.datasets.utils import load_json_arg

_MIN_TEXT_TOKENS = 2
_MAX_TORCH_SEED = 2**63 - 1

_SEQUENCE_LENGTH_STREAM = 0
_IMAGE_COUNT_STREAM = 1
_IMAGE_GEOMETRY_STREAM = 2
_TEXT_TOKEN_STREAM = 3
_IMAGE_PLACEMENT_STREAM = 4
_PIXEL_VALUE_STREAM = 5
_MODALITY_STREAM = 6

_MODALITY_INTERLEAVED = "interleaved"
_MODALITY_TEXT_ONLY = "text_only"
_MODALITY_IMAGE_ONLY = "image_only"
_SUPPORTED_MODALITIES = (_MODALITY_INTERLEAVED, _MODALITY_TEXT_ONLY, _MODALITY_IMAGE_ONLY)

_MAX_IMAGES_PER_SAMPLE = 8
_MAX_DENSITY_IMAGE_COUNT = 1024
_MAX_IMAGES_PER_1K_TOKENS = 64.0


def _normalized_categorical(
    items: Sequence[Any], weights: Sequence[Any], *, what: str
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    """Validate categorical weights, drop zero-probability items, and normalize."""
    validated_weights: list[float] = []
    for weight_index, weight in enumerate(weights):
        if not isinstance(weight, numbers.Real) or isinstance(weight, bool):
            raise ValueError(
                f"{what} weights must be finite non-negative numbers; "
                f"got {weight!r} at index {weight_index}."
            )
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{what} weights must be finite non-negative numbers; "
                f"got {weight!r} at index {weight_index}."
            )
        validated_weights.append(weight)

    max_weight = max(validated_weights)
    if max_weight <= 0:
        raise ValueError(f"{what} weights must have a finite positive sum.")
    scaled_weights = [weight / max_weight for weight in validated_weights]
    scaled_sum = math.fsum(scaled_weights)

    # Drop zero-probability items so they cannot affect the minimum supported
    # sequence length or conditional feasibility checks.
    supported = [
        (item, weight / scaled_sum) for item, weight in zip(items, scaled_weights) if weight > 0
    ]
    return tuple(item for item, _ in supported), tuple(weight for _, weight in supported)


def _seed_sequence(seed: int, idx: int, stream: int, item: int = 0) -> np.random.SeedSequence:
    """Return an access-order-independent RNG namespace for one sample stream."""
    return np.random.SeedSequence([int(seed), int(idx), int(stream), int(item)])


def _read_sequence_lengths(path: str) -> np.ndarray:
    """Read integer sequence lengths from a headerless CSV file."""
    values: list[int] = []
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"Mock varlen sequence-length file does not exist: {path}")

    with csv_path.open(newline="") as csv_file:
        for line_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row:
                raise ValueError(
                    f"Empty sequence-length row at {path}:{line_number}; "
                    "the headerless CSV must contain an integer in every field."
                )
            for column_number, raw_value in enumerate(row, start=1):
                value = raw_value.strip()
                if not value:
                    raise ValueError(
                        f"Empty sequence length at {path}:{line_number}:{column_number}; "
                        "the headerless CSV must contain an integer in every field."
                    )
                try:
                    values.append(int(value))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid sequence length {value!r} at {path}:{line_number}; "
                        "the file must contain headerless integers."
                    ) from exc

    if not values:
        raise ValueError(f"Mock varlen sequence-length file is empty: {path}")
    return np.asarray(values, dtype=np.int64)


class _SequenceLengthSampler:
    """Deterministically sample a total multimodal sequence length per index."""

    def __init__(
        self, config: dict[str, Any] | None, *, max_seq_length: int, min_seq_length: int, seed: int
    ) -> None:
        if max_seq_length < min_seq_length:
            raise ValueError(
                f"seq_length={max_seq_length} is too small for the smallest mock image; "
                f"at least {min_seq_length} tokens are required."
            )

        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.seed = seed
        self.lengths: np.ndarray | None = None

        if config is None:
            default_min = max(min_seq_length, max_seq_length // 2)
            config = {
                "mode": "distribution",
                "type": "lognormal",
                "min_seq_len": default_min,
                "max_seq_len": max_seq_length,
                "mean_seq_len": max(default_min, max_seq_length * 3 // 4),
                "lognormal_sigma": 1.1,
            }
        if not isinstance(config, dict):
            raise TypeError(f"length_config must be a dict, got {type(config).__name__}")

        self.mode = str(config.get("mode", ""))
        if self.mode == "file":
            path = config.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError("Mock varlen file mode requires a non-empty 'path'.")
            self.lengths = _read_sequence_lengths(path)
            self._validate_lengths(self.lengths, source=path)
            return

        if self.mode != "distribution":
            raise ValueError(
                f"Unsupported mock varlen mode {self.mode!r}; expected 'distribution' or 'file'."
            )
        if config.get("type") != "lognormal":
            raise ValueError(
                f"Unsupported mock varlen distribution {config.get('type')!r}; "
                "only 'lognormal' is supported."
            )

        required_fields = ("min_seq_len", "max_seq_len", "mean_seq_len", "lognormal_sigma")
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(
                "Mock varlen lognormal distribution is missing required field(s): "
                + ", ".join(missing_fields)
            )

        self.sample_min = int(config["min_seq_len"])
        self.sample_max = int(config["max_seq_len"])
        self.sample_mean = float(config["mean_seq_len"])
        self.sigma = float(config["lognormal_sigma"])
        if not self.min_seq_length <= self.sample_min <= self.sample_max <= self.max_seq_length:
            raise ValueError(
                "Mock varlen distribution must satisfy "
                f"{self.min_seq_length} <= min_seq_len <= max_seq_len <= "
                f"seq_length ({self.max_seq_length}); got "
                f"min_seq_len={self.sample_min}, max_seq_len={self.sample_max}."
            )
        if not self.sample_min <= self.sample_mean <= self.sample_max:
            raise ValueError(
                "mean_seq_len must be within [min_seq_len, max_seq_len]; got "
                f"{self.sample_mean} not in [{self.sample_min}, {self.sample_max}]."
            )
        if self.sigma < 0:
            raise ValueError(f"lognormal_sigma must be non-negative, got {self.sigma}.")
        self.mu = math.log(self.sample_mean) - self.sigma**2 / 2

    def _validate_lengths(self, lengths: np.ndarray, *, source: str) -> None:
        invalid = lengths[(lengths < self.min_seq_length) | (lengths > self.max_seq_length)]
        if invalid.size:
            raise ValueError(
                f"Sequence lengths from {source} must be in "
                f"[{self.min_seq_length}, {self.max_seq_length}]; "
                f"found {int(invalid[0])}."
            )

    def __call__(self, idx: int) -> int:
        if self.lengths is not None:
            return int(self.lengths[idx % self.lengths.size])

        rng = np.random.default_rng(_seed_sequence(self.seed, int(idx), _SEQUENCE_LENGTH_STREAM))
        sampled = rng.lognormal(mean=self.mu, sigma=self.sigma)
        return int(np.clip(sampled, self.sample_min, self.sample_max))


class _ModalitySampler:
    """Sample a per-index sample modality from a validated categorical profile.

    Modalities:

    * ``interleaved`` — image blocks embedded in text (the default).
    * ``text_only`` — no images; the whole sample is text.
    * ``image_only`` — no text tokens; the sample is one or more adjacent
      vision-start + image blocks. Every next-token target is a special
      token, so image-only samples carry no loss; keep a text-bearing
      modality in any training mix.
    """

    def __init__(self, config: dict[str, Any] | None, *, seed: int) -> None:
        self.seed = seed
        if config is None:
            self.modalities: tuple[str, ...] = (_MODALITY_INTERLEAVED,)
            self.weights: tuple[float, ...] = (1.0,)
            return
        if not isinstance(config, dict):
            raise TypeError(f"modality_config must be a dict, got {type(config).__name__}")

        mode = str(config.get("mode", "categorical"))
        if mode != "categorical":
            raise ValueError(f"Unsupported mock modality mode {mode!r}; expected 'categorical'.")
        missing_fields = [field for field in ("modalities", "weights") if field not in config]
        if missing_fields:
            raise ValueError(
                "Mock modality categorical mode is missing required field(s): "
                + ", ".join(missing_fields)
            )

        modalities = config["modalities"]
        weights = config["weights"]
        if not isinstance(modalities, (list, tuple)) or not modalities:
            raise ValueError("Mock modality 'modalities' must be a non-empty list.")
        if not isinstance(weights, (list, tuple)) or not weights:
            raise ValueError("Mock modality 'weights' must be a non-empty list.")
        if len(modalities) != len(weights):
            raise ValueError(
                "Mock modality 'modalities' and 'weights' must have the same length; "
                f"got {len(modalities)} and {len(weights)}."
            )

        validated_modalities: list[str] = []
        for modality_index, modality in enumerate(modalities):
            if modality not in _SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported mock modality {modality!r} at index {modality_index}; "
                    f"expected one of {list(_SUPPORTED_MODALITIES)}."
                )
            validated_modalities.append(str(modality))
        if len(set(validated_modalities)) != len(validated_modalities):
            raise ValueError("Mock modality 'modalities' must not contain duplicate values.")

        self.modalities, self.weights = _normalized_categorical(
            validated_modalities, weights, what="Mock modality"
        )

    @staticmethod
    def min_tokens(modality: str, *, min_count: int, min_merged_tokens: int) -> int:
        """Smallest sample length that can host one sample of *modality*."""
        if modality == _MODALITY_TEXT_ONLY:
            return _MIN_TEXT_TOKENS
        image_tokens = min_count * (1 + min_merged_tokens)
        if modality == _MODALITY_IMAGE_ONLY:
            return image_tokens
        return image_tokens + _MIN_TEXT_TOKENS

    def __call__(
        self,
        idx: int,
        *,
        sample_length: int,
        min_count: int,
        min_merged_tokens: int,
        vision_budget: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> str:
        image_min_tokens = min_count * (1 + min_merged_tokens)
        feasible = []
        budget_dropped = False
        for modality, weight in zip(self.modalities, self.weights):
            if (
                self.min_tokens(modality, min_count=min_count, min_merged_tokens=min_merged_tokens)
                > sample_length
            ):
                continue
            if (
                modality != _MODALITY_TEXT_ONLY
                and vision_budget is not None
                and image_min_tokens > vision_budget
            ):
                budget_dropped = True
                continue
            feasible.append((modality, weight))
        if budget_dropped and feasible and stats is not None:
            stats["modality_dropped_by_vision_budget"] = (
                stats.get("modality_dropped_by_vision_budget", 0) + 1
            )
        if not feasible:
            raise RuntimeError(
                f"No configured mock modality fits sample idx={idx} with length {sample_length}."
            )
        if len(feasible) == 1:
            return feasible[0][0]

        conditional_sum = math.fsum(weight for _, weight in feasible)
        probabilities = np.asarray(
            [weight / conditional_sum for _, weight in feasible], dtype=np.float64
        )
        rng = np.random.default_rng(_seed_sequence(self.seed, idx, _MODALITY_STREAM))
        selected = int(rng.choice(len(feasible), p=probabilities))
        return feasible[selected][0]


class _ImageCountSampler:
    """Sample a feasible image count from a categorical or density profile.

    ``categorical`` draws from explicit counts/weights (1 through 8).
    ``density`` draws ``Poisson(images_per_1k_tokens * L / 1000)`` clamped to
    ``[1, max_count]`` and to the sample's feasible range, so the image count
    scales with sample length and the vision-token share stays roughly
    constant across sequence-length profiles.
    """

    def __init__(self, config: dict[str, Any] | None, *, seed: int) -> None:
        self.seed = seed
        self.mode = "categorical"
        if config is None:
            self.counts = (1,)
            self.weights = (1.0,)
            return
        if not isinstance(config, dict):
            raise TypeError(f"image_count_config must be a dict, got {type(config).__name__}")

        # Counts/weights alone are a convenient one-hot test shorthand;
        # explicit profiles may still declare the categorical mode.
        mode = str(config.get("mode", "categorical"))
        if mode == "density":
            self.mode = "density"
            missing_fields = [
                field for field in ("images_per_1k_tokens", "max_count") if field not in config
            ]
            if missing_fields:
                raise ValueError(
                    "Mock image-count density mode is missing required field(s): "
                    + ", ".join(missing_fields)
                )
            density = config["images_per_1k_tokens"]
            if (
                not isinstance(density, numbers.Real)
                or isinstance(density, bool)
                or not math.isfinite(float(density))
                or not 0 < float(density) <= _MAX_IMAGES_PER_1K_TOKENS
            ):
                raise ValueError(
                    "Mock image-count 'images_per_1k_tokens' must be a finite number in "
                    f"(0, {_MAX_IMAGES_PER_1K_TOKENS}]; got {density!r}."
                )
            max_count = config["max_count"]
            if (
                not isinstance(max_count, numbers.Integral)
                or isinstance(max_count, bool)
                or not 1 <= int(max_count) <= _MAX_DENSITY_IMAGE_COUNT
            ):
                raise ValueError(
                    "Mock image-count 'max_count' must be an integer in "
                    f"[1, {_MAX_DENSITY_IMAGE_COUNT}]; got {max_count!r}."
                )
            self.images_per_1k_tokens = float(density)
            self.max_count = int(max_count)
            return
        if mode != "categorical":
            raise ValueError(
                f"Unsupported mock image-count mode {mode!r}; "
                "expected 'categorical' or 'density'."
            )
        missing_fields = [field for field in ("counts", "weights") if field not in config]
        if missing_fields:
            raise ValueError(
                "Mock image-count categorical mode is missing required field(s): "
                + ", ".join(missing_fields)
            )

        counts = config["counts"]
        weights = config["weights"]
        if not isinstance(counts, (list, tuple)) or not counts:
            raise ValueError("Mock image-count 'counts' must be a non-empty list.")
        if not isinstance(weights, (list, tuple)) or not weights:
            raise ValueError("Mock image-count 'weights' must be a non-empty list.")
        if len(counts) != len(weights):
            raise ValueError(
                "Mock image-count 'counts' and 'weights' must have the same length; "
                f"got {len(counts)} and {len(weights)}."
            )

        validated_counts: list[int] = []
        for count_index, count in enumerate(counts):
            if not isinstance(count, numbers.Integral) or isinstance(count, bool):
                raise ValueError(
                    f"Mock image counts must be integers in [1, {_MAX_IMAGES_PER_SAMPLE}]; "
                    f"got {count!r} at index {count_index}."
                )
            count = int(count)
            if count == 0:
                raise ValueError(
                    "Mock image count 0 is not supported; request image-free samples "
                    "through the modality profile (--mock-modality-config-json with "
                    "'text_only') instead."
                )
            if not 1 <= count <= _MAX_IMAGES_PER_SAMPLE:
                raise ValueError(
                    f"Mock image counts must be in [1, {_MAX_IMAGES_PER_SAMPLE}]; "
                    f"got {count} at index {count_index}."
                )
            validated_counts.append(count)
        if len(set(validated_counts)) != len(validated_counts):
            raise ValueError("Mock image-count 'counts' must not contain duplicate values.")

        self.counts, self.weights = _normalized_categorical(
            validated_counts, weights, what="Mock image-count"
        )

    @property
    def min_count(self) -> int:
        """Minimum count that has non-zero probability."""
        if self.mode == "density":
            return 1
        return min(self.counts)

    def __call__(
        self,
        idx: int,
        *,
        sample_length: int,
        min_merged_tokens: int,
        min_text_tokens: int = _MIN_TEXT_TOKENS,
        vision_budget: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> int:
        image_cost = 1 + min_merged_tokens
        budget_max = sample_length if vision_budget is None else min(sample_length, vision_budget)
        if self.mode == "density":
            max_feasible = min(
                (sample_length - min_text_tokens) // image_cost, budget_max // image_cost
            )
            if max_feasible < 1:
                raise RuntimeError(
                    f"No configured mock image count fits sample idx={idx} with length "
                    f"{sample_length}."
                )
            rng = np.random.default_rng(_seed_sequence(self.seed, idx, _IMAGE_COUNT_STREAM))
            drawn = int(rng.poisson(self.images_per_1k_tokens * sample_length / 1000.0))
            count = max(1, min(drawn, self.max_count, int(max_feasible)))
            if stats is not None and drawn > count:
                if drawn > self.max_count and count == self.max_count:
                    stats["count_clamped_by_max_count"] = (
                        stats.get("count_clamped_by_max_count", 0) + 1
                    )
                if drawn > max_feasible and count == int(max_feasible):
                    stats["count_clamped_by_feasibility"] = (
                        stats.get("count_clamped_by_feasibility", 0) + 1
                    )
            return count

        feasible = [
            (count, weight)
            for count, weight in zip(self.counts, self.weights)
            if min_text_tokens + count * image_cost <= sample_length
            and count * image_cost <= budget_max
        ]
        if not feasible:
            raise RuntimeError(
                f"No configured mock image count fits sample idx={idx} with length "
                f"{sample_length}."
            )
        if stats is not None and len(feasible) < len(self.counts):
            stats["count_categories_filtered"] = stats.get("count_categories_filtered", 0) + 1
        if len(feasible) == 1:
            return feasible[0][0]

        conditional_sum = math.fsum(weight for _, weight in feasible)
        probabilities = np.asarray(
            [weight / conditional_sum for _, weight in feasible], dtype=np.float64
        )
        rng = np.random.default_rng(_seed_sequence(self.seed, idx, _IMAGE_COUNT_STREAM))
        selected = int(rng.choice(len(feasible), p=probabilities))
        return feasible[selected][0]


@dataclass(frozen=True)
class _ImageGeometry:
    """Processed image geometry and its derived Qwen-VL patch counts."""

    height: int
    width: int
    grid_h: int
    grid_w: int
    total_patches: int
    num_merged_tokens: int


class _ImageGeometrySampler:
    """Choose an ordered tuple of processed image resolutions within a token budget."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        image_size: int,
        patch_size: int,
        spatial_merge_size: int,
        seed: int,
    ) -> None:
        if config is None:
            resolutions: Any = [[image_size, image_size]]
        else:
            if not isinstance(config, dict):
                raise TypeError(f"image_size_config must be a dict, got {type(config).__name__}")
            mode = str(config.get("mode", ""))
            if mode != "buckets":
                raise ValueError(f"Unsupported mock image-size mode {mode!r}; expected 'buckets'.")
            if "resolutions" not in config:
                raise ValueError("Mock image-size bucket mode requires a 'resolutions' field.")
            resolutions = config["resolutions"]

        if not isinstance(resolutions, (list, tuple)) or not resolutions:
            raise ValueError("Mock image-size 'resolutions' must be a non-empty list.")

        geometries: list[_ImageGeometry] = []
        alignment = patch_size * spatial_merge_size
        for bucket_index, resolution in enumerate(resolutions):
            if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
                raise ValueError(
                    "Each mock image resolution must be a [height, width] pair; "
                    f"got {resolution!r} at bucket {bucket_index}."
                )
            if any(not isinstance(value, int) or isinstance(value, bool) for value in resolution):
                raise ValueError(
                    "Mock image height and width must be integers; "
                    f"got {resolution!r} at bucket {bucket_index}."
                )

            height, width = resolution
            if height <= 0 or width <= 0:
                raise ValueError(
                    "Mock image height and width must be positive; "
                    f"got {resolution!r} at bucket {bucket_index}."
                )
            if height % alignment != 0 or width % alignment != 0:
                raise ValueError(
                    "Each mock image resolution must be divisible by "
                    f"patch_size * spatial_merge_size ({alignment}); "
                    f"got {resolution!r} at bucket {bucket_index}."
                )

            grid_h = height // patch_size
            grid_w = width // patch_size
            geometries.append(
                _ImageGeometry(
                    height=height,
                    width=width,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    total_patches=grid_h * grid_w,
                    num_merged_tokens=(grid_h // spatial_merge_size)
                    * (grid_w // spatial_merge_size),
                )
            )

        self.geometries = tuple(geometries)
        self.seed = seed
        self.min_merged_tokens = min(geometry.num_merged_tokens for geometry in self.geometries)
        self._merged_values = np.asarray(
            [geometry.num_merged_tokens for geometry in self.geometries], dtype=np.int64
        )

    def _completion_counts(self, num_images: int, budget: int) -> np.ndarray:
        """Cumulative ordered-tuple counts by merged-token budget.

        Returns ``counts`` with ``counts[m][s]`` = number of ordered
        ``m``-tuples of geometries whose merged tokens sum to at most ``s``
        (``0 <= s <= budget``). Duplicate resolutions keep their multiplicity,
        matching enumeration over the full ordered product.
        """
        tuples_by_sum = np.zeros((num_images + 1, budget + 1), dtype=np.float64)
        tuples_by_sum[0, 0] = 1.0
        for m in range(1, num_images + 1):
            for value in self._merged_values:
                if value <= budget:
                    tuples_by_sum[m, value:] += tuples_by_sum[m - 1, : budget + 1 - value]
            # Rescale each row: the sampler only consumes within-row ratios,
            # and raw ordered-tuple counts grow as |buckets|^m, which would
            # overflow for the large counts produced by the density profile.
            row_max = tuples_by_sum[m].max()
            if row_max > 0:
                tuples_by_sum[m] /= row_max
        return np.cumsum(tuples_by_sum, axis=1)

    def __call__(
        self, idx: int, *, num_images: int, max_total_merged_tokens: int
    ) -> tuple[_ImageGeometry, ...]:
        budget = int(max_total_merged_tokens)
        cumulative = self._completion_counts(num_images, budget) if budget >= 0 else None
        if cumulative is None or cumulative[num_images, budget] == 0:
            raise RuntimeError(
                f"No ordered tuple of {num_images} mock image resolution(s) fits sample "
                f"idx={idx} with capacity for {max_total_merged_tokens} merged vision tokens."
            )

        # Draw each image's bucket with probability proportional to its number
        # of feasible completions: exactly uniform over all feasible ordered
        # tuples without materializing the |buckets|^num_images product.
        rng = np.random.default_rng(
            _seed_sequence(self.seed, idx, _IMAGE_GEOMETRY_STREAM, num_images)
        )
        chosen: list[_ImageGeometry] = []
        remaining = budget
        for position in range(num_images):
            remaining_images = num_images - position - 1
            completions = np.zeros(len(self.geometries), dtype=np.float64)
            for bucket_index, value in enumerate(self._merged_values):
                if value <= remaining:
                    completions[bucket_index] = cumulative[remaining_images, remaining - value]
            probabilities = completions / completions.sum()
            bucket_index = int(rng.choice(len(self.geometries), p=probabilities))
            chosen.append(self.geometries[bucket_index])
            remaining -= int(self._merged_values[bucket_index])
        return tuple(chosen)

    def sample_one_legacy(self, idx: int, *, max_merged_tokens: int) -> _ImageGeometry:
        """Preserve the original cyclic bucket selection when count config is omitted."""
        feasible = tuple(
            geometry
            for geometry in self.geometries
            if geometry.num_merged_tokens <= max_merged_tokens
        )
        if not feasible:
            raise RuntimeError(
                f"No mock image resolution fits sample idx={idx} with capacity for "
                f"{max_merged_tokens} merged vision tokens."
            )
        bucket_index = (self.seed + int(idx)) % len(feasible)
        return feasible[bucket_index]


class MockQwen35VLVarlenDataset(Dataset):
    """Synthetic variable-length Qwen3.5-VL samples.

    Each item contains one complete, unpadded sequence. By default every item
    interleaves text with one to eight images; an optional modality profile
    additionally emits ``text_only`` items (no images, empty vision payload)
    and ``image_only`` items (no text tokens). The multimodal collator packs
    token tensors and vision payloads independently, then the model computes
    3D MRoPE position IDs from the final packed order.

    Args:
        num_samples: Virtual number of samples in the dataset.
        seq_length: Maximum total sequence length, including vision tokens.
        length_config: Optional ``MockVarlenDataset``-style configuration.
            Supported modes are a lognormal ``distribution`` and a headerless
            CSV ``file`` containing sequence lengths.
        seed: Base seed for deterministic per-index length, resolution, token, and pixel data.
        vocab_size: Padded vocabulary size used for random text tokens.
        image_token_id: Token ID for image placeholder tokens.
        video_token_id: Token ID reserved for video placeholders.
        vision_start_token_id: Token ID immediately preceding an image block.
        image_size: Fixed synthetic image height and width in pixels when
            ``image_size_config`` is omitted.
        image_size_config: Optional dynamic-resolution configuration. The
            supported form is ``{"mode":"buckets","resolutions":[[H,W], ...]}``,
            where each pair is a processed image size in pixels.
        image_count_config: Optional image-count configuration. The
            ``categorical`` mode draws from explicit ``counts`` (1 through 8)
            with arbitrary non-negative ``weights``. The ``density`` mode
            (``{"mode":"density","images_per_1k_tokens":1.4,"max_count":64}``)
            draws ``Poisson(images_per_1k_tokens * L / 1000)`` clamped to
            ``[1, max_count]`` and to the feasible range, so the image count
            scales with sample length and the vision-token share stays
            roughly constant across sequence-length profiles. When omitted,
            every image-bearing sample contains exactly one image.
        modality_config: Optional categorical modality mix, e.g.
            ``{"mode":"categorical","modalities":["interleaved","text_only",
            "image_only"],"weights":[83,15,2]}``. Modalities that cannot fit a
            sample's length are dropped and the rest renormalized per index.
            When omitted, every sample is interleaved (legacy behavior).
            ``text_only`` items keep the exact sampled length and carry empty
            ``pixel_values``/``image_grid_thw`` tensors. ``image_only`` items
            treat the sampled length as an upper budget: the realized length is
            ``N + sum(V_j)`` and every next-token target is masked, so they
            contribute no loss.
        max_vision_tokens: Optional absolute per-sample cap on vision tokens
            (image placeholders plus vision-start tokens). At
            ``micro_batch_size=1`` this bounds the vision payload of every
            microbatch exactly, making attention-workspace usage a
            configuration-time guarantee (raw patches <= 4 x cap).
        max_vision_fraction: Optional relative per-sample cap: vision tokens
            <= ``fraction * L``. Composable with ``max_vision_tokens`` (the
            tighter bound wins). Counts/geometries that no longer fit are
            dropped and renormalized per index; when no image fits, the
            modality renormalizes to ``text_only`` (if enabled).
        image_placement: ``center`` keeps image blocks at the middle text gap;
            ``uniform`` independently samples each image's text gap.
        patch_size: Spatial patch size used by the vision encoder.
        temporal_patch_size: Temporal patch size folded into each pixel row's
            feature dimension. A still image has grid ``T=1``.
        spatial_merge_size: Spatial patch-merger factor.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 1024,
        length_config: dict[str, Any] | None = None,
        seed: int = 1234,
        vocab_size: int = 248320,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        image_size: int = 224,
        image_size_config: dict[str, Any] | None = None,
        image_count_config: dict[str, Any] | None = None,
        modality_config: dict[str, Any] | None = None,
        max_vision_tokens: int | None = None,
        max_vision_fraction: float | None = None,
        image_placement: str = "center",
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
    ) -> None:
        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}.")
        if patch_size <= 0 or temporal_patch_size <= 0:
            raise ValueError("patch_size and temporal_patch_size must be positive.")
        if spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}.")
        if not isinstance(image_placement, str) or image_placement not in {"center", "uniform"}:
            raise ValueError(
                f"image_placement must be 'center' or 'uniform', got {image_placement!r}."
            )
        if image_size_config is None:
            if image_size <= 0:
                raise ValueError(f"image_size must be positive, got {image_size}.")
            if image_size % patch_size != 0:
                raise ValueError(
                    f"image_size={image_size} must be divisible by patch_size={patch_size}."
                )

        self.num_samples = num_samples
        self.seq_length = seq_length
        self.seed = seed
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_size = image_size
        self.image_size_config = image_size_config
        self.image_count_config = image_count_config
        self.modality_config = modality_config
        self.image_placement = image_placement
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size

        self.image_geometry_sampler = _ImageGeometrySampler(
            image_size_config,
            image_size=image_size,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            seed=seed,
        )
        self.image_count_sampler = _ImageCountSampler(image_count_config, seed=seed)
        self.modality_sampler = _ModalitySampler(modality_config, seed=seed)
        self.pixel_dim = 3 * temporal_patch_size * patch_size * patch_size

        if max_vision_tokens is not None:
            if (
                not isinstance(max_vision_tokens, numbers.Integral)
                or isinstance(max_vision_tokens, bool)
                or int(max_vision_tokens) < 1
            ):
                raise ValueError(
                    f"max_vision_tokens must be a positive integer, got {max_vision_tokens!r}."
                )
            max_vision_tokens = int(max_vision_tokens)
        if max_vision_fraction is not None:
            if (
                not isinstance(max_vision_fraction, numbers.Real)
                or isinstance(max_vision_fraction, bool)
                or not 0 < float(max_vision_fraction) <= 1
            ):
                raise ValueError(
                    "max_vision_fraction must be a number in (0, 1], "
                    f"got {max_vision_fraction!r}."
                )
            max_vision_fraction = float(max_vision_fraction)
        self.max_vision_tokens = max_vision_tokens
        self.max_vision_fraction = max_vision_fraction

        image_bearing_modalities = [
            modality
            for modality in self.modality_sampler.modalities
            if modality != _MODALITY_TEXT_ONLY
        ]
        image_min_tokens = self.image_count_sampler.min_count * (
            1 + self.image_geometry_sampler.min_merged_tokens
        )
        if image_bearing_modalities:
            if max_vision_tokens is not None and max_vision_tokens < image_min_tokens:
                raise ValueError(
                    f"max_vision_tokens={max_vision_tokens} cannot host the smallest "
                    f"configured image payload ({image_min_tokens} vision tokens)."
                )
            if (
                max_vision_fraction is not None
                and int(max_vision_fraction * seq_length) < image_min_tokens
            ):
                raise ValueError(
                    f"max_vision_fraction={max_vision_fraction} cannot host the smallest "
                    f"configured image payload ({image_min_tokens} vision tokens) even at "
                    f"seq_length={seq_length}."
                )

        # Per-index observability for budget/clamp interactions; incremented
        # by the samplers and __getitem__, reset via reset_budget_stats().
        self.budget_stats: dict[str, int] = {}

        min_num_images = self.image_count_sampler.min_count
        min_seq_length = min(
            self._min_length_for_modality(modality, min_num_images, image_min_tokens)
            for modality in self.modality_sampler.modalities
        )

        special_ids = {image_token_id, video_token_id, vision_start_token_id}
        if len(special_ids) != 3:
            raise ValueError("image, video, and vision-start token IDs must be distinct.")
        if 0 in special_ids:
            raise ValueError("Multimodal token ID 0 is reserved for multimodal packing padding.")
        if vocab_size <= 1 or any(
            token_id < 0 or token_id >= vocab_size for token_id in special_ids
        ):
            raise ValueError(
                f"All multimodal token IDs must be in [0, vocab_size={vocab_size}); "
                f"got {sorted(special_ids)}."
            )
        self.special_ids = special_ids
        self.safe_text_token_id = next(
            (token_id for token_id in range(1, vocab_size) if token_id not in special_ids), None
        )
        if self.safe_text_token_id is None:
            raise ValueError("vocab_size does not contain a usable non-special text token ID.")

        self.length_sampler = _SequenceLengthSampler(
            length_config, max_seq_length=seq_length, min_seq_length=min_seq_length, seed=seed
        )

    def __len__(self) -> int:
        return self.num_samples

    def _min_length_for_modality(self, modality: str, min_count: int, image_min_tokens: int) -> int:
        """Smallest sample length hosting *modality* under the vision budget."""
        base = _ModalitySampler.min_tokens(
            modality,
            min_count=min_count,
            min_merged_tokens=self.image_geometry_sampler.min_merged_tokens,
        )
        if modality != _MODALITY_TEXT_ONLY and self.max_vision_fraction is not None:
            base = max(base, math.ceil(image_min_tokens / self.max_vision_fraction))
        return base

    def _vision_budget(self, sample_length: int) -> int:
        """Per-sample cap on vision tokens (image placeholders + vision starts)."""
        budget = sample_length
        if self.max_vision_tokens is not None:
            budget = min(budget, self.max_vision_tokens)
        if self.max_vision_fraction is not None:
            budget = min(budget, int(self.max_vision_fraction * sample_length))
        return budget

    def reset_budget_stats(self) -> None:
        """Clear the per-index budget/clamp counters."""
        self.budget_stats.clear()

    def _generator(self, idx: int, stream: int, item: int = 0) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        seed_state = _seed_sequence(self.seed, idx, stream, item).generate_state(1, dtype=np.uint64)
        generator.manual_seed(int(seed_state[0]) % _MAX_TORCH_SEED)
        return generator

    def _image_gaps(self, idx: int, *, text_length: int, num_images: int) -> list[int]:
        """Choose insertion gaps in the original text token coordinates."""
        if self.image_placement == "center":
            return [text_length // 2] * num_images
        return [
            int(
                np.random.default_rng(
                    _seed_sequence(self.seed, idx, _IMAGE_PLACEMENT_STREAM, image_idx)
                ).integers(0, text_length + 1)
            )
            for image_idx in range(num_images)
        ]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.num_samples == 0:
            raise IndexError("Cannot index an empty MockQwen35VLVarlenDataset.")
        sample_idx = int(idx) % self.num_samples
        sample_length = self.length_sampler(sample_idx)
        vision_budget = self._vision_budget(sample_length)
        modality = self.modality_sampler(
            sample_idx,
            sample_length=sample_length,
            min_count=self.image_count_sampler.min_count,
            min_merged_tokens=self.image_geometry_sampler.min_merged_tokens,
            vision_budget=vision_budget,
            stats=self.budget_stats,
        )
        min_text_tokens = 0 if modality == _MODALITY_IMAGE_ONLY else _MIN_TEXT_TOKENS
        if modality == _MODALITY_TEXT_ONLY:
            num_images = 0
            geometries: tuple[_ImageGeometry, ...] = ()
        elif self.image_count_config is None:
            num_images = 1
            geometries = (
                self.image_geometry_sampler.sample_one_legacy(
                    sample_idx,
                    max_merged_tokens=min(sample_length - 1 - min_text_tokens, vision_budget - 1),
                ),
            )
        else:
            num_images = self.image_count_sampler(
                sample_idx,
                sample_length=sample_length,
                min_merged_tokens=self.image_geometry_sampler.min_merged_tokens,
                min_text_tokens=min_text_tokens,
                vision_budget=vision_budget,
                stats=self.budget_stats,
            )
            geometries = self.image_geometry_sampler(
                sample_idx,
                num_images=num_images,
                max_total_merged_tokens=min(
                    sample_length - num_images - min_text_tokens, vision_budget - num_images
                ),
            )
        total_merged_tokens = sum(geometry.num_merged_tokens for geometry in geometries)
        if num_images and num_images + total_merged_tokens >= 0.95 * vision_budget:
            self.budget_stats["budget_saturated"] = self.budget_stats.get("budget_saturated", 0) + 1
        # An image-only sample treats the sampled length as an upper budget:
        # discrete image geometries cannot fill it exactly without text.
        if modality == _MODALITY_IMAGE_ONLY:
            text_length = 0
        else:
            text_length = sample_length - num_images - total_merged_tokens
        if text_length < min_text_tokens:
            raise RuntimeError(
                f"Mock sample idx={sample_idx} reserved only {text_length} text tokens; "
                f"at least {min_text_tokens} are required."
            )

        text_tokens = torch.randint(
            1,
            self.vocab_size,
            (text_length,),
            dtype=torch.long,
            generator=self._generator(sample_idx, stream=_TEXT_TOKEN_STREAM),
        )
        for special_id in self.special_ids:
            text_tokens[text_tokens == special_id] = self.safe_text_token_id

        gaps = self._image_gaps(sample_idx, text_length=text_length, num_images=num_images)
        image_order = sorted(range(num_images), key=lambda image_idx: (gaps[image_idx], image_idx))
        ordered_geometries = tuple(geometries[image_idx] for image_idx in image_order)

        token_chunks: list[torch.Tensor] = []
        text_cursor = 0
        for image_idx in image_order:
            gap = gaps[image_idx]
            geometry = geometries[image_idx]
            token_chunks.extend(
                [
                    text_tokens[text_cursor:gap],
                    torch.tensor([self.vision_start_token_id], dtype=torch.long),
                    torch.full(
                        (geometry.num_merged_tokens,), self.image_token_id, dtype=torch.long
                    ),
                ]
            )
            text_cursor = gap
        token_chunks.append(text_tokens[text_cursor:])
        input_ids = torch.cat(token_chunks)

        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        target_is_special = labels == -100
        for special_id in self.special_ids:
            target_is_special |= labels == special_id
        labels[target_is_special] = -100
        loss_mask[target_is_special] = 0.0

        if num_images:
            pixels_by_image = tuple(
                torch.randn(
                    geometry.total_patches,
                    self.pixel_dim,
                    generator=self._generator(
                        sample_idx, stream=_PIXEL_VALUE_STREAM, item=image_idx
                    ),
                )
                for image_idx, geometry in enumerate(geometries)
            )
            pixel_values = torch.cat(
                [pixels_by_image[image_idx] for image_idx in image_order], dim=0
            )
            image_grid_thw = torch.tensor(
                [[1, geometry.grid_h, geometry.grid_w] for geometry in ordered_geometries],
                dtype=torch.long,
            )
        else:
            pixel_values = torch.empty((0, self.pixel_dim))
            image_grid_thw = torch.empty((0, 3), dtype=torch.long)

        expected_length = (
            num_images + total_merged_tokens if modality == _MODALITY_IMAGE_ONLY else sample_length
        )
        if input_ids.numel() != expected_length or input_ids.numel() > sample_length:
            raise RuntimeError(
                f"Generated {input_ids.numel()} tokens for expected length {expected_length} "
                f"(budget {sample_length})."
            )
        vision_start_positions = torch.where(input_ids == self.vision_start_token_id)[0]
        if vision_start_positions.numel() != num_images:
            raise RuntimeError("Vision-start token count does not match the sampled image count.")
        if int((input_ids == self.image_token_id).sum().item()) != total_merged_tokens:
            raise RuntimeError("Image placeholder count does not match the merged vision grid.")
        expected_patches = sum(geometry.total_patches for geometry in ordered_geometries)
        if pixel_values.shape != (expected_patches, self.pixel_dim):
            raise RuntimeError("Pixel rows do not match the raw patch grids.")
        if image_grid_thw.shape != (num_images, 3):
            raise RuntimeError("Image grid rows do not match the sampled image count.")
        for vision_start, geometry in zip(vision_start_positions.tolist(), ordered_geometries):
            image_block = input_ids[
                vision_start + 1 : vision_start + 1 + geometry.num_merged_tokens
            ]
            if image_block.numel() != geometry.num_merged_tokens or not torch.all(
                image_block == self.image_token_id
            ):
                raise RuntimeError(
                    "Each vision-start token must be followed by its complete image block."
                )
        has_loss_target = bool(loss_mask.sum().item() > 0)
        if modality == _MODALITY_IMAGE_ONLY:
            if has_loss_target:
                raise RuntimeError("Image-only samples must not carry loss targets.")
        elif not has_loss_target:
            raise RuntimeError("Text-bearing samples must keep at least one loss target.")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def train_valid_test_varlen_datasets_provider(
    train_val_test_num_samples: Sequence[int],
) -> tuple[MockQwen35VLVarlenDataset, MockQwen35VLVarlenDataset, MockQwen35VLVarlenDataset]:
    """Provide variable-length mock train, validation, and test datasets."""
    from megatron.training import get_args

    args = get_args()
    if getattr(args, "use_varlen_dataset", False):
        raise ValueError(
            "The multimodal mock_varlen provider is incompatible with --use-varlen-dataset; "
            "vision payloads are packed by multimodal_dev.forward_step instead."
        )
    if getattr(args, "sequence_packing_scheduler", None) is not None:
        raise ValueError(
            "The multimodal mock_varlen provider is incompatible with "
            "--sequence-packing-scheduler; vision payloads are packed by "
            "multimodal_dev.forward_step instead."
        )
    uses_hybridep = (
        getattr(args, "use_packed_sequence", False)
        and getattr(args, "moe_token_dispatcher_type", None) == "flex"
        and getattr(args, "moe_flex_dispatcher_backend", None) == "hybridep"
    )
    if uses_hybridep and not getattr(args, "moe_hybridep_pad_variable_tokens", False):
        raise ValueError(
            "The multimodal mock_varlen provider requires "
            "--moe-hybridep-pad-variable-tokens with packed THD + HybridEP; "
            "locally packed token counts can differ across the HybridEP group."
        )
    if not getattr(args, "use_vanilla_collate_fn", False):
        raise ValueError(
            "The multimodal mock_varlen provider requires --use-vanilla-collate-fn "
            "so variable-length samples remain a list until multimodal packing."
        )

    total_seq_length = int(getattr(args, "total_seq_length", 1024))
    model_seq_length = getattr(args, "seq_length", None)
    if model_seq_length is None or total_seq_length != int(model_seq_length):
        raise ValueError(
            "The multimodal mock_varlen provider requires --total-seq-length to equal "
            f"--seq-length; got total_seq_length={total_seq_length}, "
            f"seq_length={model_seq_length}."
        )

    length_config = load_json_arg(getattr(args, "varlen_mock_dataset_config_json", None))
    image_size_config = load_json_arg(getattr(args, "mock_image_size_config_json", None))
    image_count_config = load_json_arg(getattr(args, "mock_image_count_config_json", None))
    modality_config = load_json_arg(getattr(args, "mock_modality_config_json", None))
    kwargs = dict(
        seq_length=total_seq_length,
        length_config=length_config,
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        image_size=getattr(args, "image_size", 224),
        image_size_config=image_size_config,
        image_count_config=image_count_config,
        modality_config=modality_config,
        max_vision_tokens=getattr(args, "mock_max_vision_tokens", None),
        max_vision_fraction=getattr(args, "mock_max_vision_fraction", None),
        image_placement=getattr(args, "mock_image_placement", "center"),
    )
    seed = int(getattr(args, "seed", 1234))

    train_ds = MockQwen35VLVarlenDataset(
        num_samples=train_val_test_num_samples[0], seed=seed, **kwargs
    )
    if set(train_ds.modality_sampler.modalities) == {_MODALITY_IMAGE_ONLY}:
        raise ValueError(
            "The multimodal mock_varlen modality profile must keep a text-bearing "
            "modality: image_only samples mask every next-token target, so an "
            "image_only-only mix would train on zero loss tokens."
        )
    val_ds = MockQwen35VLVarlenDataset(
        num_samples=train_val_test_num_samples[1], seed=seed + 1, **kwargs
    )
    test_ds = MockQwen35VLVarlenDataset(
        num_samples=train_val_test_num_samples[2], seed=seed + 2, **kwargs
    )
    return train_ds, val_ds, test_ds
