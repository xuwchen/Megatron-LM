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

## Variable-Length Mock Data

`mock_varlen` generates one complete, unpadded Qwen3.5-VL sequence per
dataset item: text interleaved with images by default, plus optional
`text_only` (no images) and `image_only` (no text tokens) items. The
identity collator is required because token lengths differ and each item's
token and vision payloads must stay associated; `--use-packed-sequence`
selects the packed-THD packer (padded BSHD otherwise):

```bash
torchrun --nproc_per_node=8 examples/multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock_varlen \
    --seq-length 32768 \
    --total-seq-length 32768 \
    --use-vanilla-collate-fn \
    --use-packed-sequence \
    --max-seqlen-per-dp-cp-rank 32768 \
    --pad-packed-seq-alignment 128 \
    --varlen-mock-dataset-config-json \
      '{"mode":"distribution","type":"lognormal","min_seq_len":1024,"max_seq_len":32768,"mean_seq_len":8192,"lognormal_sigma":1.1}' \
    --mock-modality-config-json \
      '{"mode":"categorical","modalities":["interleaved","text_only","image_only"],"weights":[83,15,2]}' \
    --mock-image-count-config-json \
      '{"mode":"density","images_per_1k_tokens":1.4,"max_count":64}' \
    --mock-image-size-config-json \
      '{"mode":"buckets","resolutions":[[224,224],[448,448],[672,448],[448,672],[896,672],[1120,896]]}' \
    --mock-image-placement uniform \
    ... # other Megatron model and training arguments
```

Each dataset item has exactly these fields:

| Field | Per-sample shape | Meaning |
|-------|------------------|---------|
| `input_ids` | `[L_i]` | Text tokens plus `N_i` complete image-placeholder blocks |
| `labels` | `[L_i]` | Shifted next-token labels; ignored targets are `-100` |
| `loss_mask` | `[L_i]` | Float mask aligned with `labels` |
| `pixel_values` | `[sum(P_j), D]` | Ordered flattened raw patches for all images |
| `image_grid_thw` | `[N_i, 3]` | Ordered `(T, H, W)` patch grids |

Configuration summary (every JSON option accepts inline JSON or a file
path; omitting an option preserves the legacy fixed one-image behavior):

- `--varlen-mock-dataset-config-json`: bounded lognormal `distribution`
  lengths (sampled *truncated*: out-of-window mass is renormalized into the
  window, so there are no endpoint spikes and `mean_seq_len` is the
  post-truncation expectation), or a headerless CSV `file` of exact integer
  lengths.
- `--mock-modality-config-json`: categorical mix of `interleaved`,
  `text_only`, and `image_only` items. `text_only` items carry zero-row
  `pixel_values`/`image_grid_thw`; `image_only` items treat the sampled
  length as an upper budget and carry zero loss tokens, so a mix whose only
  modality is `image_only` is rejected.
- `--mock-image-count-config-json`: `categorical` explicit counts in
  `[1, 8]`, or `density` (`Poisson(images_per_1k_tokens * L / 1000)`
  clamped to `[1, max_count]` and to the feasible range) so image counts
  scale with item length and the vision-token share stays stable across
  sequence-length profiles.
- `--mock-image-size-config-json`: processed `[height, width]` buckets,
  each divisible by `patch_size * spatial_merge_size`; a feasible ordered
  geometry tuple is sampled per item. An optional `weights` list (same
  length) weights the buckets: each tuple's probability is proportional to
  the product of its members' weights over the feasible set.
- `--mock-image-placement`: `center` (legacy) or `uniform` text-gap
  placement of each complete vision-start + image block.
- `--mock-max-vision-tokens` / `--mock-max-vision-fraction`: per-sample
  caps on vision tokens (placeholders + vision starts). At
  `micro_batch_size=1` a microbatch is one sample, so an absolute cap
  bounds every microbatch's raw patches at `4 x cap` — the vision
  attention-workspace footprint becomes a configuration-time guarantee.
  Counts/geometries that no longer fit are renormalized per index; short
  samples that physically cannot reach an absolute cap are unaffected.
- `--max-vision-patches-per-microbatch` / `--max-vision-patches-per-image`:
  packer-level fail-fast guards checked before the TP broadcast; violations
  raise with the actual payload, the limit, and the offending geometry
  instead of surfacing as an opaque CUDA OOM.

Profiles that cannot fit a sampled item length are conditioned per index:
infeasible modalities/counts/geometries are dropped and the remaining
weights renormalized. Token blocks, `image_grid_thw` rows, and
`pixel_values` slices always use the same image order, and generation is
deterministic and access-order independent per index.

### packed_window mode

`--varlen-mock-dataset-config-json '{"mode":"packed_window",...}'` selects
the production-shaped generator: documents (disjoint short/long
truncated-lognormal text lengths, per-document `text_only` probability
`p_text`, per-document `Gamma`-mixed Poisson image density) concatenate
into a stream that is sliced into fixed `seq_length`-token windows. One
dataset item is one full window; a sixth per-sample field `seq_lens`
carries the per-segment lengths (`sum == seq_length`), and the packed-THD
packer splices those segments into `cu_seqlens` with independent per-
segment CP/SP alignment padding. Image atoms never cross window lines
(spill/fill construction with an explicitly counted `boundary_fill`
budget), each segment's final position carries no loss target, and the
window-level modality/count/budget flags above are rejected as obsolete —
counts and modality mixes are emergent from the document layer.
`doc_length.long_component_text_token_share` must be set explicitly per
recipe, and `micro_batch_size` must be 1. Requires
`--use-packed-sequence` for multi-segment windows (BSHD has no segment
representation).

Do **not** combine with `--use-varlen-dataset` or
`--sequence-packing-scheduler` (text-only core contract; neither carries
ragged vision tensors). Packed THD + HybridEP flex dispatch requires
`--moe-hybridep-pad-variable-tokens`. When `--pad-packed-seq-alignment` is
set, real samples keep their CP/SP alignment and the packed tail is
represented as one ordinary dummy THD sequence (token rows only — no
pixels, grids, or loss); `--max-seqlen-per-dp-cp-rank` is the CP-local
target for `alignment=max`. An image-free microbatch still runs the vision
tower once on a minimal zero-weighted dummy image so every rank produces
vision grads for bucketed grad synchronization.

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
