# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""
import argparse


def add_multimodal_args(parser):
    """Add multimodal-specific arguments to the Megatron argument parser."""
    group = parser.add_argument_group("Multimodal", "Multimodal model arguments")

    group.add_argument(
        "--model-arch",
        type=str,
        default="qwen35_vl",
        help="Model architecture. Available: qwen35_vl",
    )
    group.add_argument(
        "--model-variant",
        type=str,
        default="proxy",
        help="Model variant (size). E.g. proxy, 9b, 397b_a17b",
    )
    group.add_argument(
        "--dataset-provider",
        type=str,
        default="mock",
        help=(
            "Dataset provider: mock, mock_varlen, or cord_v2. Every provider "
            "emits the multimodal sample contract; a pure-text corpus is "
            "mock_varlen with text_only_document_probability 1.0."
        ),
    )
    group.add_argument(
        "--multimodal-varlen-mock-dataset-config-json",
        type=str,
        default=None,
        help=(
            "packed_window corpus + image-size config for --dataset-provider "
            "mock_varlen. Accepts inline JSON or a JSON-file path. OPTIONAL: "
            "omitted entirely, the frozen parity-calibrated profile applies; "
            "a partial JSON overrides exactly the top-level keys it carries "
            "(mode, doc_length, text_only_document_probability, "
            "image_poisson_rate_per_1k_text_tokens, image_density_gamma_shape, "
            "image_sizes, max_boundary_fill_fraction, plan_pool_windows, "
            "plan_seed) and unknown keys are rejected. doc_length also accepts the string "
            '"context_scaled": an opt-in coverage policy whose long-document '
            "token share grows +5pp per doubling of seq-length (0.25 at 4K "
            "to 0.60 at 512K; longer windows need explicit components). Distinct from the core "
            "--varlen-mock-dataset-config-json flag, which belongs to the "
            "text-side --use-varlen-dataset datasets. See the packed_window "
            "README section for the schema and the defaults' provenance."
        ),
    )
    group.add_argument(
        "--image-token-id", type=int, default=248056, help="Token ID for image placeholder tokens"
    )
    group.add_argument(
        "--image-size", type=int, default=224, help="Image size (height and width) for mock data"
    )
    group.add_argument(
        # Migration trap: folded into --multimodal-varlen-mock-dataset-config-json
        # as the top-level "image_sizes" key. The mock_varlen provider raises
        # an instructive error when this is set.
        "--mock-image-size-config-json",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--total-seq-length",
        type=int,
        default=None,
        help=(
            "Total sequence length for mock data. Default: inherits "
            "--seq-length for mock_varlen packed windows (which are exactly "
            "seq_length tokens by construction); 1024 for the fixed-shape "
            "providers."
        ),
    )
    group.add_argument(
        "--image-seq-length", type=int, default=256, help="Number of image tokens in mock data"
    )
    group.add_argument(
        "--vision-num-layers",
        type=int,
        default=None,
        help=("Override for vision backbone depth. " "Useful for proxy perf runs."),
    )
    group.add_argument(
        "--hf-processor-path",
        type=str,
        default=None,
        help=(
            "HuggingFace processor path for real VLM datasets " "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)"
        ),
    )
    group.add_argument(
        "--recompute-vision",
        action="store_true",
        default=False,
        help=(
            "Enable full activation recomputation for vision encoder layers. "
            "Uses uniform method and recomputes every layer. "
            "Independent of the decoder --recompute-* flags."
        ),
    )
    group.add_argument(
        "--use-packed-sequence",
        action="store_true",
        default=False,
        help=("Pack variable-length sequences into THD format to eliminate " "padding waste."),
    )
    group.add_argument(
        "--pad-packed-seq-by-appending-dummy-seq",
        dest="pad_packed_seq_by_appending_dummy_seq",
        action="store_true",
        default=True,
        help=(
            "Positive alias so recipe YAMLs can state the dummy-THD-tail "
            "contract explicitly. The setting is already the core default "
            "and the packed_window path requires it on."
        ),
    )
    group.add_argument(
        "--max-vision-patches-per-microbatch",
        type=int,
        default=None,
        help=(
            "Fail fast when one microbatch's vision payload exceeds this many "
            "raw patches (checked before TP broadcast). The vision tower's "
            "packed attention workspace scales stepwise with total raw patches, "
            "so exceeding the memory envelope otherwise surfaces as an opaque "
            "CUDA OOM. Unset by default."
        ),
    )
    group.add_argument(
        "--max-vision-patches-per-image",
        type=int,
        default=None,
        help=(
            "Fail fast when any single image exceeds this many raw patches "
            "(checked before TP broadcast). For --dataset-provider mock_varlen "
            "the default is derived from the bucket table (the largest "
            "drawable bucket's raw-patch count, ignoring weight-0 buckets — "
            "an exact invariant of the data); unset otherwise."
        ),
    )
    group.add_argument(
        "--use-vanilla-collate-fn",
        action="store_true",
        default=False,
        help=("Use vanilla collate function to collate the data."),
    )

    return parser
