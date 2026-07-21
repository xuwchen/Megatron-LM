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

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(idx)]))
        sampled = rng.lognormal(mean=self.mu, sigma=self.sigma)
        return int(np.clip(sampled, self.sample_min, self.sample_max))


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
    """Deterministically choose a feasible processed image resolution per index."""

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

    def __call__(self, idx: int, *, max_merged_tokens: int) -> _ImageGeometry:
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

        # Consecutive indices cycle through the feasible buckets from a
        # seed-dependent offset. This is deterministic and access-order independent.
        bucket_index = (self.seed + int(idx)) % len(feasible)
        return feasible[bucket_index]


class MockQwen35VLVarlenDataset(Dataset):
    """Synthetic variable-length Qwen3.5-VL samples.

    Each item contains one complete, unpadded image-text sequence. The
    multimodal collator packs token tensors and vision payloads independently,
    then the model computes 3D MRoPE position IDs from the final packed order.

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
        self.pixel_dim = 3 * temporal_patch_size * patch_size * patch_size
        min_seq_length = 1 + self.image_geometry_sampler.min_merged_tokens + _MIN_TEXT_TOKENS

        special_ids = {image_token_id, video_token_id, vision_start_token_id}
        if len(special_ids) != 3:
            raise ValueError("image, video, and vision-start token IDs must be distinct.")
        if 0 in special_ids:
            raise ValueError(
                "Multimodal token ID 0 is reserved for multimodal packing padding."
            )
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

    def _generator(self, idx: int, stream: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + idx * 2 + stream) % _MAX_TORCH_SEED)
        return generator

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.num_samples == 0:
            raise IndexError("Cannot index an empty MockQwen35VLVarlenDataset.")
        sample_idx = int(idx) % self.num_samples
        sample_length = self.length_sampler(sample_idx)
        geometry = self.image_geometry_sampler(
            sample_idx, max_merged_tokens=sample_length - 1 - _MIN_TEXT_TOKENS
        )
        text_length = sample_length - geometry.num_merged_tokens - 1

        text_tokens = torch.randint(
            1,
            self.vocab_size,
            (text_length,),
            dtype=torch.long,
            generator=self._generator(sample_idx, stream=0),
        )
        for special_id in self.special_ids:
            text_tokens[text_tokens == special_id] = self.safe_text_token_id

        prefix_length = text_length // 2
        input_ids = torch.cat(
            [
                text_tokens[:prefix_length],
                torch.tensor([self.vision_start_token_id], dtype=torch.long),
                torch.full((geometry.num_merged_tokens,), self.image_token_id, dtype=torch.long),
                text_tokens[prefix_length:],
            ]
        )

        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        loss_mask = torch.ones(sample_length, dtype=torch.float32)
        target_is_special = labels == -100
        for special_id in self.special_ids:
            target_is_special |= labels == special_id
        labels[target_is_special] = -100
        loss_mask[target_is_special] = 0.0

        pixel_values = torch.randn(
            geometry.total_patches, self.pixel_dim, generator=self._generator(sample_idx, stream=1)
        )
        image_grid_thw = torch.tensor([[1, geometry.grid_h, geometry.grid_w]], dtype=torch.long)

        if input_ids.numel() != sample_length:
            raise RuntimeError(
                f"Generated {input_ids.numel()} tokens for requested length {sample_length}."
            )
        if int((input_ids == self.image_token_id).sum().item()) != geometry.num_merged_tokens:
            raise RuntimeError("Image placeholder count does not match the merged vision grid.")

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
    kwargs = dict(
        seq_length=total_seq_length,
        length_config=length_config,
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        image_size=getattr(args, "image_size", 224),
        image_size_config=image_size_config,
    )
    seed = int(getattr(args, "seed", 1234))

    train_ds = MockQwen35VLVarlenDataset(
        num_samples=train_val_test_num_samples[0], seed=seed, **kwargs
    )
    val_ds = MockQwen35VLVarlenDataset(
        num_samples=train_val_test_num_samples[1], seed=seed + 1, **kwargs
    )
    test_ds = MockQwen35VLVarlenDataset(
        num_samples=train_val_test_num_samples[2], seed=seed + 2, **kwargs
    )
    return train_ds, val_ds, test_ds
