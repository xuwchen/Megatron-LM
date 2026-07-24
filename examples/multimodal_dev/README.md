# multimodal_dev — Standalone Multimodal Training

Standalone, model-agnostic training entry point for multimodal
vision-language models built on Megatron-Core (FSDP + EP).

## Directory Structure

```
multimodal_dev/
├── pretrain_multimodal.py   # Training entry point (model-agnostic)
├── forward_step.py          # Forward step, TP broadcast, loss computation
├── arguments.py             # Multimodal CLI arguments
├── data/
│   ├── mock.py              # Fixed-length mock data for end-to-end testing
│   ├── mock_varlen.py       # Variable-length mock image-text data
│   └── cord_v2.py           # CORD-V2 receipt-OCR data provider
├── models/
│   ├── __init__.py          # MODEL_REGISTRY — central model registry
│   ├── base.py              # MultimodalModel base class (vision encoder + GPTModel)
│   └── qwen35_vl/           # Qwen3.5-VL architecture
│       ├── factory.py       # Factory functions for pretrain entry point
│       ├── model.py         # Qwen35VLModel (MRoPE, vision encoder wiring)
│       ├── configuration.py # TransformerConfig builders and constants
│       ├── specs.py         # Layer spec builders (hybrid attention, ViT)
│       ├── mrope.py         # 3D MRoPE position ID computation
│       └── vision_encoder.py# ViT encoder (patch embed, merger, RoPE)
└── scripts/                 # Launch scripts (torchrun, Slurm)
```

## Quick Start

```bash
torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock \
    ... # other Megatron args (--num-layers, --hidden-size, etc.)
```

## Variable-Length Mock Data (packed_window)

`mock_varlen` generates deterministic fixed-length packed THD windows from
a configurable short/long document mixture (document-count weights), with
text-only documents and length-scaled multimodal image density. One
dataset item is one full `--seq-length` training window sliced from a mock
document stream: documents concatenate and are cut at window boundaries,
so a window carries one or more document segments (`seq_lens`). Image
blocks (`vision_start` + merged placeholder tokens) are indivisible and
never cross window lines. Fixed-shape single-image data is served by
`--dataset-provider mock` instead.

Runs out of the box: with no dataset config at all, the frozen
**parity-calibrated profile** applies (its window-level statistics match a
production 4K packed-training reference; provenance below).

```bash
torchrun --nproc_per_node=8 examples/multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock_varlen \
    --seq-length 4096 \
    --micro-batch-size 1 \
    --use-vanilla-collate-fn \
    --use-packed-sequence \
    --pad-packed-seq-alignment 128 \
    ... # other Megatron model and training arguments
```

To deviate, pass a partial JSON overriding exactly the top-level keys it
carries (omitted keys keep the calibrated defaults; unknown keys are
rejected). Example — a customer-hypothesis coverage profile (95/5
short/long by document count) with a custom bucket table:

```bash
    --multimodal-varlen-mock-dataset-config-json \
      '{"doc_length":{"components":[{"name":"short","weight":95,"min":1024,"max":2048,"mean":1536,"sigma":0.3},{"name":"long","weight":5,"min":65536,"max":131072,"mean":98304,"sigma":0.4}]},"image_sizes":{"resolutions":[[224,224],[448,448],[672,448]],"weights":[3,2,1]}}'
```

Each dataset item has exactly these fields:

| Field | Per-sample shape | Meaning |
|-------|------------------|---------|
| `input_ids` | `[S]` | Always exactly `seq_length` tokens |
| `labels` | `[S]` | Next-token targets; `-100` at each segment's final position (no cross-document prediction) and at image/vision-start targets |
| `loss_mask` | `[S]` | Float mask aligned with `labels` |
| `pixel_values` | `[sum(P_j), D]` | Ordered flattened raw patches for all images |
| `image_grid_thw` | `[N, 3]` | Ordered `(T, H, W)` patch grids |
| `seq_lens` | `[num_segments]` | Logical per-segment lengths, `sum == S` |

Configuration:

- `--multimodal-varlen-mock-dataset-config-json`: the single, optional
  generator config (inline JSON or a JSON-file path). Omitted keys fall
  back to the calibrated parity profile (`PARITY_PROFILE` in
  `data/mock_varlen.py`, locked by tests); unknown keys are rejected so a
  typo cannot silently run a different distribution. Distinct from the
  core `--varlen-mock-dataset-config-json`, which belongs to the
  text-side `--use-varlen-dataset` datasets. Top-level keys:
  - `doc_length.components`: truncated-lognormal components with
    **document-count** `weight`s (normalized internally — corpus
    descriptions like "95% short / 5% long" map directly); each has
    `min`/`max`/`mean` (post-truncation)/`sigma` (0 = constant length).
    `doc_length` also accepts the string `"context_scaled"`, an opt-in
    coverage POLICY (an engineering interpolation, not a fitted law):
    the long-document token share grows +5 percentage points per
    doubling of the window — 0.25 at 4K (the parity-era coverage
    hypothesis), 0.50 at 128K (the qualified 128K recipe), capped at
    0.60 from 512K (ProLong-inspired); the long component itself stays
    the corpus's real long-doc population at every S, and windows
    beyond 512K require explicit components. Explicit components always
    win.
  - `text_only_document_probability`: the **exact** text-only document
    probability (interleaved documents draw a zero-truncated count and
    always carry at least one image).
  - `image_poisson_rate_per_1k_text_tokens`: the LATENT per-document
    Poisson rate of interleaved documents — the realized zero-truncated
    density is slightly higher (lambda/(1-e^-lambda), ~+8.5% at the
    parity profile) and short documents floor at one image, so measure
    realized densities with the simulator. Per-document rates are
    Gamma-mixed with `image_density_gamma_shape` (1.0 = exponential).
  - `image_sizes`: processed `[height, width]` resolution buckets (each
    divisible by `patch_size * spatial_merge_size`) with optional
    categorical `weights`; per-image sizes are drawn from this set.
  - `max_boundary_fill_fraction`: `boundary_fill` tokens emitted at
    window lines are ordinary text with normal loss; their fraction is
    bounded by this ceiling (enforced at construction).
  - `plan_pool_windows`: bounds the pre-built plan corpus independently
    of the virtual dataset length. Default `"auto"` =
    `max(2048, ceil(2**26 / seq_length))`: at least 64Mi plan tokens at
    any `seq_length`, so rare mixture components (e.g. the 0.518% long
    documents) stay statistically represented in the layout pool instead
    of drifting with the pool seed. An explicit integer is an upper
    bound, clamped by the split's sample count (e.g. pin the pool to one
    exact pass for a worst-window sweep).
  - `plan_seed`: seeds the window LAYOUT (default 1234, the
    calibration-snapshot seed), independently of `--seed`. The workload
    shape is a constant of the profile — training seeds vary token and
    pixel content only — so throughput/memory numbers stay comparable
    across seeds; override the key to study layout variance.

  Window-level statistics — segments per window, image counts,
  vision-token share — are emergent and measured, not configured.

  Default provenance (every value is traceable): the FITTED values
  (`text_only_document_probability` 0.735,
  `image_poisson_rate_per_1k_text_tokens` 1.76,
  `image_density_gamma_shape` 0.46, the bucket weights) come from a
  coordinate fit against production 4K packed-window statistics,
  validated on an independent seed; the DERIVED values (component
  weights 99.482/0.518 = the exact solution of a 25% long-doc text-token
  share given the post-truncation means 2048/131072; the fill ceiling;
  the pool size) follow from stated round targets; the remaining round
  numbers (component means/windows/sigmas, the resolution ladder) are
  STRUCTURAL HYPOTHESES whose window-level consequences the parity fit
  validates. See the comment block above `PARITY_PROFILE`.
- `--max-vision-patches-per-microbatch` / `--max-vision-patches-per-image`:
  hard packer-level fail-fast guards checked before the TP broadcast;
  violations raise with the actual payload, the limit, and the offending
  geometry instead of surfacing as an opaque CUDA OOM. The per-image
  guard defaults to the largest drawable (weight > 0) bucket's
  raw-patch count (an exact
  invariant of the bucket table); the per-microbatch guard stays
  explicit because it states a hardware memory envelope.
- `--total-seq-length`: inherited from `--seq-length` when omitted
  (packed windows are exactly `seq_length` tokens by construction).

The packed-THD packer splices each sample's `seq_lens` segments into
`cu_seqlens` with independent per-segment CP/SP alignment padding (real
padding tokens after every internal segment, so the tensor layout matches
`cu_seqlens_padded`); the padded BSHD layout rejects multi-segment samples.
`micro_batch_size` must be 1 — one item already is a full window — and
`--use-packed-sequence` plus the identity collator
(`--use-vanilla-collate-fn`) are required (both enforced by the provider).
The dataset mirrors `--max-vision-patches-per-microbatch` as a per-window
budget checked from plan geometry BEFORE pixels are materialized, so
over-budget windows fail without paying the multi-GiB host allocation;
over-budget windows fail fast by design until the chunked/streaming
vision runtime raises the envelope. Do **not**
combine with `--use-varlen-dataset` or `--sequence-packing-scheduler`.
Packed THD + HybridEP flex dispatch requires
`--moe-hybridep-pad-variable-tokens`. An image-free microbatch still runs
the vision tower once on a minimal zero-weighted dummy image so every rank
produces vision grads for bucketed grad synchronization.

Generation is deterministic and access-order independent per index; the
CPU calibration simulator consumes the same plan generator, so window
statistics can be measured without materializing tokens or pixels.

## Checkpoint Conversion (HF → Megatron-FSDP DTensor)

Convert a HuggingFace release to a Megatron-FSDP DTensor checkpoint via
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) before
pretraining from pretrained weights.

### Setup

Clone Bridge and pin its `3rdparty/Megatron-LM` submodule to this branch:

```bash
git clone --recurse-submodules https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge/3rdparty/Megatron-LM
git remote add wplf https://github.com/wplf/Megatron-LM.git
git fetch wplf feat/qwen35-vl-example
git checkout feat/qwen35-vl-example
cd ../..
```

### Convert

Single 8×GPU node, EP=8 / TP=CP=1; substitute any Qwen3.5 variant for
`--hf-model`:

```bash
PYTHONPATH=./src:./3rdparty/Megatron-LM/ \
  torchrun --nproc_per_node=8 \
  examples/conversion/mfsdp/convert_checkpoints_fsdp.py import \
  --hf-model Qwen/Qwen3.5-35B-A3B \
  --megatron-path ${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B-fsdp \
  --ckpt-format fsdp_dtensor \
  --ep 8
```

HF weights are auto-fetched on first run via `huggingface_hub`. Adjust
`--tp` / `--cp` / `--ep` to match the training topology (must satisfy
`WORLD_SIZE % (TP*CP*EP) == 0`).

### Output

```
${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B-fsdp/
├── iter_0000000/
│   ├── __0_0.distcp .. __7_0.distcp   # FSDP DTensor shards, one per rank (~18 GB each for 35B-A3B)
│   ├── .metadata
│   ├── run_config.yaml
│   └── train_state.pt
├── latest_checkpointed_iteration.txt
└── latest_train_state.pt
```

### Bridge dependency

Requires
[NVIDIA-NeMo/Megatron-Bridge#3987](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3987)
(skip tokenizer save). Without that fix the checkpoint is still written
correctly but the script exits non-zero after save with
`AttributeError: 'TokenizerConfig' object has no attribute 'make_vocab_size_divisible_by'`
against this branch's `megatron.core.tokenizers.utils.build_tokenizer`.

## Architecture

`pretrain_multimodal.py` is **model-agnostic**. All model-specific logic
is delegated to factory functions registered in `MODEL_REGISTRY`
(`models/__init__.py`). The entry point handles only generic concerns:

- Building `language_config` from Megatron CLI args
- Constructing `vision_config` via the registry
- Applying vision recompute and dtype propagation
- Routing to model and dataset factories

The `forward_step` is also model-agnostic — it uses the model's
`compute_position_ids()` method polymorphically and passes a standard
batch dict.

## Adding a New Model Architecture

Adding a new model (e.g. `llava_next`) requires **no changes** to
`pretrain_multimodal.py` or `forward_step.py`. Follow these steps:

### Step 1 — Create the model package

```
multimodal_dev/models/llava_next/
├── __init__.py
├── factory.py          # Required: factory functions
├── configuration.py    # Vision/language TransformerConfig builders
├── model.py            # Model class (subclass MultimodalModel)
├── specs.py            # Layer spec builders
└── vision_encoder.py   # Vision encoder (if custom)
```

### Step 2 — Implement factory functions

Create `factory.py` with up to three functions:

```python
# models/llava_next/factory.py

def post_language_config(language_config, args):
    """(Optional) Mutate language_config with model-specific fields."""
    # e.g. language_config.some_field = value
    pass

def set_vision_flops_metadata(args, language_config, vision_config):
    """(Optional) Set vision FLOPs metadata on args."""
    args.count_vision_model_flops = True
    args.vision_flops_variant = "llava_next"
    # ... set dimension fields for FLOPs calculation

def build_model(args, language_config, vision_config, **kwargs):
    """(Required) Build and return the complete model instance."""
    from .model import LlavaNextModel
    from .specs import get_llava_next_language_spec

    language_spec = get_llava_next_language_spec(
        config=language_config,
        vp_stage=kwargs.get("vp_stage", None),
        pp_rank=None,
    )
    return LlavaNextModel(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        # ... model-specific args
    )
```

### Step 3 — Register in `MODEL_REGISTRY`

Add an entry in `models/__init__.py`:

```python
from multimodal_dev.models.llava_next.configuration import (
    get_llava_next_vision_config,
)
from multimodal_dev.models.llava_next.factory import (
    build_model as _build_llava_next_model,
    post_language_config as _llava_next_post_language_config,
    set_vision_flops_metadata as _llava_next_vision_flops,
)

MODEL_REGISTRY["llava_next"] = {
    "model_factory_fn": _build_llava_next_model,           # required
    "vision_config_fn": get_llava_next_vision_config,      # required
    "post_language_config_fn": _llava_next_post_language_config,  # optional
    "vision_flops_fn": _llava_next_vision_flops,           # optional
    "dataset_providers": {                                  # optional
        "mock": "multimodal_dev.data.llava_mock.train_valid_test_datasets_provider",
    },
}
```

### Step 4 — (Optional) Add a dataset provider

Create a dataset module under `data/` if the model needs custom data
preprocessing. The provider function signature is:

```python
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Return (train_dataset, val_dataset, test_dataset)."""
    ...
```

Register it in the `dataset_providers` dict of the registry entry.
Providers can be either direct callables or dotted import path strings
(resolved lazily at runtime).

### Step 5 — Launch

```bash
torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \
    --model-arch llava_next \
    --dataset-provider mock \
    ...
```

## Registry Entry Reference

| Field | Required | Signature |
|-------|----------|-----------|
| `model_factory_fn` | Yes | `(args, language_config, vision_config, **kwargs) -> MegatronModule` |
| `vision_config_fn` | Yes | `(num_layers_override=None) -> TransformerConfig` |
| `post_language_config_fn` | No | `(language_config, args) -> None` |
| `vision_flops_fn` | No | `(args, language_config, vision_config) -> None` |
| `dataset_providers` | No | `Dict[str, str \| callable]` |
