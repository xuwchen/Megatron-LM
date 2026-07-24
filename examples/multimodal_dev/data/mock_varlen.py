# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed-window mock image-text data for Qwen3.5-VL training.

Generates deterministic fixed-length packed THD windows from a
configurable short/long document mixture (document-count weights), with
text-only documents and length-scaled multimodal image density.

One dataset item is one full ``seq_length``-token training window sliced
from a mock document stream planned by
:class:`~examples.multimodal_dev.data.packed_window_plan.PackedWindowPlanGenerator`
(the same plan source the CPU calibration simulator uses). This module is
a token/pixel adapter over that plan; window-level statistics (segments
per window, image counts, vision share) are emergent from the document
layer and are measured, not configured.

The generic text-only ``MockVarlenDataset`` cannot transport ragged vision
payloads through the core packing scheduler, so this provider keeps the
raw per-sample contract and leaves multimodal packing to
``multimodal_dev.forward_step.pack_or_pad_batch``. Fixed-shape single-image
scenarios are served by ``--dataset-provider mock`` instead.
"""

import numbers
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.data.packed_window_plan import (
    PARITY_PROFILE,
    context_scaled_doc_length,
)
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from megatron.training.datasets.utils import load_json_arg

_MAX_TORCH_SEED = 2**63 - 1

# Auto plan-pool sizing target: >= 2^26 (64Mi) plan tokens regardless of
# seq_length, so rare mixture components stay statistically represented in
# the layout pool (see PARITY_PROFILE provenance in packed_window_plan.py).
AUTO_PLAN_POOL_TOKENS = 1 << 26

_TEXT_TOKEN_STREAM = 3
_PIXEL_VALUE_STREAM = 5


def _seed_sequence(seed: int, idx: int, stream: int, item: int = 0) -> np.random.SeedSequence:
    """Return an access-order-independent RNG namespace for one sample stream."""
    return np.random.SeedSequence([int(seed), int(idx), int(stream), int(item)])


class PackedWindowQwen35VLDataset(Dataset):
    """Fixed-length packed windows sliced from a mock document stream.

    Contract (six fields): ``input_ids``/``labels``/``loss_mask`` of shape
    ``[seq_length]``, ``pixel_values [total_raw_patches, pixel_dim]``,
    ``image_grid_thw [num_images, 3]``, and ``seq_lens [num_segments]``
    with ``seq_lens.sum() == seq_length``. Labels are next-token targets;
    each segment's final position is ``-100`` (no cross-document
    prediction), as are targets that land on image or vision-start tokens.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        seq_length: int,
        window_config: dict[str, Any],
        seed: int = 1234,
        vocab_size: int = 248320,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        image_size_config: dict[str, Any] | None = None,
        max_raw_patches_per_window: int | None = None,
        streaming_pixels: bool = False,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
    ) -> None:
        from examples.multimodal_dev.data.packed_window_plan import PackedWindowPlanGenerator

        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
        if patch_size <= 0 or temporal_patch_size <= 0 or spatial_merge_size <= 0:
            raise ValueError(
                "patch_size, temporal_patch_size, and spatial_merge_size must be positive."
            )
        if (
            not isinstance(image_size_config, dict)
            or not image_size_config.get("resolutions")
            or not set(image_size_config) <= {"resolutions", "weights"}
        ):
            raise ValueError(
                "packed_window requires image_size_config = "
                '{"resolutions": [[H, W], ...]} with optional "weights" (no '
                f"other keys); got {image_size_config!r}."
            )

        block = patch_size * spatial_merge_size
        grids: list[tuple[int, int, int]] = []
        merged_tokens: list[int] = []
        raw_patches: list[int] = []
        for index, resolution in enumerate(image_size_config["resolutions"]):
            if (
                not isinstance(resolution, (list, tuple))
                or len(resolution) != 2
                or not all(
                    isinstance(side, numbers.Integral) and not isinstance(side, bool)
                    for side in resolution
                )
                or not all(int(side) > 0 for side in resolution)
            ):
                raise ValueError(
                    f"Bucket resolution at index {index} must be exactly two positive "
                    f"integers [height, width]; got {resolution!r}."
                )
            height, width = int(resolution[0]), int(resolution[1])
            if height % block or width % block:
                raise ValueError(
                    f"Bucket resolution {height}x{width} must be divisible by "
                    f"patch_size*spatial_merge_size={block}."
                )
            grid_h, grid_w = height // patch_size, width // patch_size
            grids.append((1, grid_h, grid_w))
            merged_tokens.append((grid_h // spatial_merge_size) * (grid_w // spatial_merge_size))
            raw_patches.append(grid_h * grid_w)
        if "weights" not in image_size_config:
            weights = [1.0] * len(grids)
        else:
            # Present-but-invalid must fail loudly: `[]`, null, or wrong types
            # must never silently degrade to uniform weights. Zero entries are
            # legal (they disable a bucket); the kernel remains the single
            # authority for finite / non-negative / positive-sum.
            weights = image_size_config["weights"]
            if (
                not isinstance(weights, (list, tuple))
                or not weights
                or any(isinstance(w, bool) or not isinstance(w, numbers.Real) for w in weights)
            ):
                raise ValueError(
                    "Bucket 'weights', when present, must be a non-empty list of "
                    f"numbers (zeros allowed; the sum must be positive); got {weights!r}."
                )
        if len(weights) != len(grids):
            raise ValueError(
                f"Bucket 'weights' must match 'resolutions' in length; got "
                f"{len(weights)} weights for {len(grids)} resolutions."
            )

        special_ids = {image_token_id, video_token_id, vision_start_token_id}
        # Token ID 0 is reserved for packing padding; a special ID of 0 could
        # be miscounted as an image placeholder after collate padding.
        if any(not 0 < token_id < vocab_size for token_id in special_ids):
            raise ValueError(
                f"All multimodal token IDs must be in [1, vocab_size={vocab_size}); "
                f"got {sorted(special_ids)}."
            )
        self.safe_text_token_id = next(
            (token_id for token_id in range(1, vocab_size) if token_id not in special_ids), None
        )
        if self.safe_text_token_id is None:
            raise ValueError("vocab_size does not contain a usable non-special text token ID.")

        self.num_samples = int(num_samples)
        self.seq_length = int(seq_length)
        self.seed = int(seed)
        self.vocab_size = int(vocab_size)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.special_ids = special_ids
        self.grids = grids
        self.bucket_weights = [float(weight) for weight in weights]
        self.pixel_dim = 3 * temporal_patch_size * patch_size * patch_size
        if max_raw_patches_per_window is not None and int(max_raw_patches_per_window) <= 0:
            raise ValueError(
                "max_raw_patches_per_window must be a positive integer or None "
                f"(0 does not silently disable it); got {max_raw_patches_per_window!r}."
            )
        self.max_raw_patches_per_window = (
            int(max_raw_patches_per_window) if max_raw_patches_per_window is not None else None
        )
        self.streaming_pixels = bool(streaming_pixels)
        # The plan pool bounds construction time and memory independently of
        # the virtual dataset length Megatron requests for the full training
        # schedule: indices wrap onto the pool for the window LAYOUT while
        # token/pixel content stays keyed by the virtual index.
        window_config = dict(window_config)
        pool_windows = window_config.pop("plan_pool_windows", "auto")
        if pool_windows == "auto":
            # >= AUTO_PLAN_POOL_TOKENS plan tokens at any S: enough expected
            # long documents in the pool (~110 under the parity mixture) that
            # the realized long-doc token share stays near nominal instead of
            # drifting with the pool seed; the floor keeps long-context pools
            # at the proven 2048-window cost bound.
            pool_windows = max(2048, -(-AUTO_PLAN_POOL_TOKENS // seq_length))
        elif isinstance(pool_windows, bool) or not isinstance(pool_windows, int):
            raise ValueError(
                "plan_pool_windows must be the string 'auto' or a positive "
                f"integer, got {pool_windows!r}."
            )
        if pool_windows <= 0:
            raise ValueError(f"plan_pool_windows must be positive, got {pool_windows}.")
        self.plan_pool_windows = min(pool_windows, num_samples) if num_samples else 0
        # The window LAYOUT is seeded by plan_seed (profile default 1234, the
        # calibration-snapshot seed), independently of the training seed: the
        # workload shape — segment structure, image placement, per-window
        # payloads — is a constant of the profile, while --seed varies
        # token/pixel CONTENT only. Finite pools realize heavy-tailed
        # statistics (Gamma-mixed image density) with visible seed-to-seed
        # variance, so a floating layout would make throughput/memory numbers
        # incomparable across seeds.
        plan_seed = window_config.pop("plan_seed", 1234)
        if isinstance(plan_seed, bool) or not isinstance(plan_seed, int):
            raise ValueError(f"plan_seed must be an integer, got {plan_seed!r}.")
        self.plan_seed = int(plan_seed)
        self.plan = (
            PackedWindowPlanGenerator(
                seq_length=seq_length,
                num_windows=self.plan_pool_windows,
                seed=self.plan_seed,
                config=window_config,
                bucket_merged_tokens=merged_tokens,
                bucket_raw_patches=raw_patches,
                bucket_weights=weights,
            )
            if num_samples > 0
            else None
        )

    def __len__(self) -> int:
        return self.num_samples

    def _generator(self, idx: int, stream: int, item: int = 0) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        seed_state = _seed_sequence(self.seed, idx, stream, item).generate_state(1, dtype=np.uint64)
        generator.manual_seed(int(seed_state[0]) % _MAX_TORCH_SEED)
        return generator

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.num_samples == 0:
            raise IndexError("Cannot index an empty PackedWindowQwen35VLDataset.")
        idx = int(idx) % self.num_samples
        window = self.plan.window(idx % self.plan_pool_windows)
        # Enforce the patch budget from the plan geometry BEFORE any pixel
        # tensor is materialized: the packer-level guard alone cannot prevent
        # the DataLoader/host-memory peak of a heavy window (a 128K tail
        # window is multiple GiB of fp32 pixels).
        total_raw_patches = sum(atom.raw_patches for atom in window.atoms)
        if (
            self.max_raw_patches_per_window is not None
            and total_raw_patches > self.max_raw_patches_per_window
        ):
            raise ValueError(
                f"Window {idx} carries {total_raw_patches} raw vision patches, exceeding "
                f"max_raw_patches_per_window={self.max_raw_patches_per_window} "
                f"({len(window.atoms)} images). Long-window profiles require chunked "
                "vision-encoder execution (Phase B) before raising this budget."
            )

        input_ids = torch.randint(
            1,
            self.vocab_size,
            (self.seq_length,),
            dtype=torch.long,
            generator=self._generator(idx, stream=_TEXT_TOKEN_STREAM),
        )
        for special_id in self.special_ids:
            input_ids[input_ids == special_id] = self.safe_text_token_id
        for atom in window.atoms:
            input_ids[atom.offset] = self.vision_start_token_id
            input_ids[atom.offset + 1 : atom.offset + 1 + atom.merged_tokens] = self.image_token_id

        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        # No cross-document prediction: the last position of every segment
        # has no target inside its own document.
        boundary = 0
        for _, segment_length in window.segments:
            boundary += segment_length
            labels[boundary - 1] = -100
        target_is_special = labels == -100
        for special_id in self.special_ids:
            target_is_special |= labels == special_id
        labels[target_is_special] = -100
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        loss_mask[target_is_special] = 0.0

        if window.atoms:
            image_grid_thw = torch.tensor(
                [self.grids[atom.bucket_index] for atom in window.atoms], dtype=torch.long
            )
        else:
            image_grid_thw = torch.empty((0, 3), dtype=torch.long)
        if self.streaming_pixels:
            # Synthetic-streaming profile: geometry only. The model
            # materializes chunk inputs as views into its noise pool.
            pixel_values = None
        elif window.atoms:
            # One preallocated buffer, filled per image: no per-image tensor
            # list + concat, so the host peak is the payload itself rather
            # than twice the payload.
            pixel_values = torch.empty((total_raw_patches, self.pixel_dim), dtype=torch.float32)
            row = 0
            for ordinal, atom in enumerate(window.atoms):
                pixel_values[row : row + atom.raw_patches].normal_(
                    generator=self._generator(idx, stream=_PIXEL_VALUE_STREAM, item=ordinal)
                )
                row += atom.raw_patches
        else:
            pixel_values = torch.empty((0, self.pixel_dim), dtype=torch.float32)

        seq_lens = torch.tensor([length for _, length in window.segments], dtype=torch.long)
        if int(seq_lens.sum().item()) != self.seq_length:
            raise RuntimeError(
                f"Window {idx} segment lengths sum to {int(seq_lens.sum().item())}; "
                f"expected {self.seq_length}."
            )

        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "image_grid_thw": image_grid_thw,
            "seq_lens": seq_lens,
        }
        if pixel_values is not None:
            sample["pixel_values"] = pixel_values
        return sample


def largest_drawable_bucket_patches(dataset: "PackedWindowQwen35VLDataset") -> int:
    """Raw patches (T*H*W) of the largest bucket that can actually be drawn.

    Zero-weight buckets are disabled and must not gate feasibility or lift
    guard bounds. The kernel guarantees a positive weight sum whenever a plan
    was built; guard against the degenerate zero-sample split explicitly so
    an all-zero table cannot reach an empty max().
    """
    sizes = [
        grid_t * grid_h * grid_w
        for (grid_t, grid_h, grid_w), weight in zip(dataset.grids, dataset.bucket_weights)
        if weight > 0
    ]
    if not sizes:
        raise ValueError("Bucket weights leave no drawable bucket (all zero).")
    return max(sizes)


def resolve_varlen_config(spec: str | None, *, seq_length: int) -> dict[str, Any]:
    """Shallow-merge a user config over the calibrated parity profile.

    ``spec`` is the raw --multimodal-varlen-mock-dataset-config-json value
    (inline JSON or a JSON-file path) and may be None: the profile applies
    unchanged. A partial JSON overrides exactly the top-level keys it
    carries; unknown keys are rejected so a typo cannot silently fall back
    to a default and run a distribution the user did not write.
    """
    user = load_json_arg(spec)
    if user is None:
        user = {}
    if not isinstance(user, dict):
        raise ValueError(
            "--multimodal-varlen-mock-dataset-config-json must be a JSON "
            f"object; got {type(user).__name__}."
        )
    if user.get("mode", "packed_window") != "packed_window":
        raise ValueError(
            'mock_varlen supports only {"mode": "packed_window"} in '
            "--multimodal-varlen-mock-dataset-config-json (the legacy "
            "distribution/file sample modes were removed; use "
            "--dataset-provider mock for fixed-shape data)."
        )
    unknown = set(user) - set(PARITY_PROFILE)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in "
            "--multimodal-varlen-mock-dataset-config-json; allowed top-level "
            f"keys: {sorted(PARITY_PROFILE)}. Omitted keys fall back to the "
            "calibrated parity profile, so only overridden keys need to be "
            "present."
        )
    config = {**PARITY_PROFILE, **user}
    doc_length = config["doc_length"]
    if doc_length == "context_scaled":
        # Opt-in coverage policy: the long-document token share scales
        # with the window (+5pp per doubling, capped; see
        # packed_window_plan.context_scaled_doc_length). Explicit
        # components always win by simply not using the string form.
        doc_length = context_scaled_doc_length(seq_length)
        config["doc_length"] = doc_length
        from megatron.training import print_rank_0

        short_weight = doc_length["components"][0]["weight"]
        long_weight = doc_length["components"][1]["weight"]
        print_rank_0(
            f"[mock_varlen] context_scaled doc_length at seq_length={seq_length}: "
            f"component weights {short_weight}/{long_weight} "
            "(short/long, document-count)"
        )
    if not isinstance(doc_length, dict) or set(doc_length) != {"components"}:
        raise ValueError(
            'doc_length must be {"components": [...]} (a document-count '
            'weighted mixture) or the string "context_scaled"; got '
            f"{doc_length!r}."
        )
    # image_sizes structure is validated once, by the dataset constructor
    # (the single authority for bucket validation).
    return config


def train_valid_test_varlen_datasets_provider(
    train_val_test_num_samples,
) -> tuple[PackedWindowQwen35VLDataset, PackedWindowQwen35VLDataset, PackedWindowQwen35VLDataset]:
    """Provide packed-window mock train, validation, and test datasets."""
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
    if not getattr(args, "use_packed_sequence", False):
        raise ValueError(
            "The multimodal mock_varlen packed_window provider requires "
            "--use-packed-sequence: windows carry multiple document segments "
            "(seq_lens) and the padded BSHD layout has no segment representation."
        )
    uses_hybridep = (
        getattr(args, "moe_token_dispatcher_type", None) == "flex"
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

    model_seq_length = getattr(args, "seq_length", None)
    if model_seq_length is None:
        raise ValueError("The multimodal mock_varlen provider requires --seq-length.")
    total_seq_length = getattr(args, "total_seq_length", None)
    if total_seq_length is None:
        # packed_window items are full seq_length-token windows by
        # construction, so the total length is not an independent choice.
        total_seq_length = int(model_seq_length)
    elif int(total_seq_length) != int(model_seq_length):
        raise ValueError(
            "The multimodal mock_varlen provider requires --total-seq-length to equal "
            f"--seq-length (or to be omitted); got total_seq_length={total_seq_length}, "
            f"seq_length={model_seq_length}."
        )
    total_seq_length = int(total_seq_length)

    if getattr(args, "varlen_mock_dataset_config_json", None) is not None:
        raise ValueError(
            "The multimodal mock_varlen provider no longer reads the core "
            "--varlen-mock-dataset-config-json flag (it belongs to the "
            "text-side --use-varlen-dataset datasets). Move the config to "
            "--multimodal-varlen-mock-dataset-config-json; omitted keys fall "
            "back to the calibrated parity profile."
        )
    if getattr(args, "mock_image_size_config_json", None) is not None:
        raise ValueError(
            "--mock-image-size-config-json was folded into "
            "--multimodal-varlen-mock-dataset-config-json as the top-level "
            '"image_sizes" key: {"image_sizes": {"resolutions": [...], '
            '"weights": [...]}} (the inner "mode": "buckets" tag is gone). '
            "Omit the key entirely for the calibrated 15-bucket table."
        )
    config = resolve_varlen_config(
        getattr(args, "multimodal_varlen_mock_dataset_config_json", None),
        seq_length=total_seq_length,
    )
    image_size_config = config["image_sizes"]
    window_config = {
        key: value for key, value in config.items() if key not in ("mode", "image_sizes")
    }

    streaming_pixels = bool(getattr(args, "mock_synthetic_streaming_pixels", False))
    if streaming_pixels and int(getattr(args, "vision_encoder_chunk_patches", 0) or 0) <= 0:
        raise ValueError(
            "--mock-synthetic-streaming-pixels requires "
            "--vision-encoder-chunk-patches > 0 (the noise pool holds one chunk)."
        )
    micro_batch_size = int(getattr(args, "micro_batch_size", 1) or 1)
    if micro_batch_size != 1:
        raise ValueError(
            "packed_window mode requires micro_batch_size == 1 for training and "
            f"evaluation: one item already is a full {total_seq_length}-token "
            f"window; got micro_batch_size={micro_batch_size}."
        )

    kwargs = dict(
        seq_length=total_seq_length,
        window_config=window_config,
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        image_size_config=image_size_config,
        # Mirror the packer guard at the dataset so over-budget windows fail
        # before pixels are materialized on the host.
        max_raw_patches_per_window=getattr(args, "max_vision_patches_per_microbatch", None),
        streaming_pixels=streaming_pixels,
    )
    seed = int(getattr(args, "seed", 1234))
    datasets = tuple(
        PackedWindowQwen35VLDataset(
            num_samples=train_val_test_num_samples[split], seed=seed + split, **kwargs
        )
        for split in range(3)
    )

    if streaming_pixels:
        # The noise pool holds exactly one chunk of raw-patch rows, and
        # images are indivisible: a bucket larger than the chunk budget can
        # never stream. Fail at startup instead of probabilistically at the
        # first heavy draw deep in a model forward.
        largest_bucket_patches = largest_drawable_bucket_patches(datasets[0])
        chunk_patches = int(getattr(args, "vision_encoder_chunk_patches", 0) or 0)
        if largest_bucket_patches > chunk_patches:
            raise ValueError(
                "--mock-synthetic-streaming-pixels: the largest image bucket "
                f"needs {largest_bucket_patches} raw patches but the noise pool "
                f"holds one chunk of --vision-encoder-chunk-patches="
                f"{chunk_patches}; raise the chunk budget or shrink the bucket "
                "table (images are indivisible, so an oversized bucket can "
                "never stream)."
            )

    if getattr(args, "max_vision_patches_per_image", None) is None:
        # The per-image guard is a true invariant, not a tunable: atom sizes
        # come from the same (now fully validated) bucket table, so its exact
        # upper bound is the largest drawable (weight > 0) bucket's raw-patch
        # count. Resolve it once so the packer-side check (forward_step) sees
        # a concrete bound even when no explicit cap was configured.
        args.max_vision_patches_per_image = largest_drawable_bucket_patches(datasets[0])
    return datasets
