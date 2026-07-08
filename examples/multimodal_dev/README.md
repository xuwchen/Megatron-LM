# multimodal_dev — Standalone Multimodal Training

Standalone, model-agnostic training entry point for multimodal
vision-language models built on Megatron-Core (FSDP + EP).

## Directory Structure

```
multimodal_dev/
├── pretrain_multimodal.py   # Training entry point (model-agnostic)
├── forward_step.py          # Forward step, TP broadcast, loss computation
├── arguments.py             # Multimodal CLI arguments
├── VARLEN_MOCK_DATASET.md   # Text and multimodal varlen design guide
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

For the text-only baseline introduced by
[NVIDIA/Megatron-LM#4832](https://github.com/NVIDIA/Megatron-LM/pull/4832), and the
multimodal extension described here, see
[Mock Variable-Length Datasets: Text and Multimodal](VARLEN_MOCK_DATASET.md).
The guide covers the contracts, packing diagrams, and supported scope.

`mock_varlen` generates one complete, unpadded Qwen3.5-VL image-text
sequence per dataset item. Use the identity collator because token lengths
differ and the token and vision payloads must remain associated per sample.
Add `--use-packed-sequence` when the intended decoder layout is packed THD:

```bash
torchrun --nproc_per_node=8 examples/multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock_varlen \
    --seq-length 32768 \
    --total-seq-length 32768 \
    --use-vanilla-collate-fn \
    --use-packed-sequence \
    --linear-cp-mode chunkwise \
    --max-seqlen-per-dp-cp-rank 32768 \
    --pad-packed-seq-alignment 128 \
    --pad-packed-seq-by-appending-dummy-seq \
    --varlen-mock-dataset-config-json \
      '{"mode":"distribution","type":"lognormal","min_seq_len":1024,"max_seq_len":32768,"mean_seq_len":8192,"lognormal_sigma":1.1}' \
    --mock-image-size-config-json \
      '{"mode":"buckets","resolutions":[[224,224],[224,448],[448,224],[448,448]]}' \
    ... # other Megatron model and training arguments
```

The provider reuses `--varlen-mock-dataset-config-json`. It accepts either
the lognormal `distribution` form above or a headerless CSV containing one or
more integer sequence lengths:

```bash
--varlen-mock-dataset-config-json \
  '{"mode":"file","path":"/path/to/sequence_lengths.csv"}'
```

Dynamic image resolutions are optional. `--mock-image-size-config-json`
accepts inline JSON or a JSON-file path with processed `[height, width]`
buckets. Each dimension must be divisible by
`patch_size * spatial_merge_size`. The sampler deterministically cycles
through the buckets that fit the sampled `L_i`; without this option,
`--image-size` remains the fixed square fallback.

`--use-vanilla-collate-fn` is required for `mock_varlen`.
`--use-packed-sequence` selects the multimodal THD packer; without it,
`pack_or_pad_batch` produces a padded BSHD batch instead. Do **not** add
`--use-varlen-dataset` or `--sequence-packing-scheduler`: those options
select the text-only core scheduler, whose sample schema and communication path
do not carry the ragged vision tensors. `--sft` is not required to select
`mock_varlen`; an existing SFT recipe may retain it for its loss-reporting
semantics as long as the text-only `--use-varlen-dataset` flag is absent.

```text
MockQwen35VLVarlenDataset
    |  one unpadded five-field sample
    v
identity collate (--use-vanilla-collate-fn)
    |  list[dict[str, Tensor]]
    v
pack_or_pad_batch
    |-- padded BSHD (default)
    `-- packed THD + PackedSeqParams (--use-packed-sequence)
             |  optional final-tail alignment + dummy THD sequence
             v
Qwen3.5-VL: MRoPE -> vision encode -> masked scatter
             |
             v
context-parallel split -> language decoder
```

Each dataset item has exactly these fields:

| Field | Per-sample shape | Meaning |
|-------|------------------|---------|
| `input_ids` | `[L_i]` | Text tokens plus one complete image-placeholder block |
| `labels` | `[L_i]` | Shifted next-token labels; ignored targets are `-100` |
| `loss_mask` | `[L_i]` | Float mask aligned with `labels` |
| `pixel_values` | `[P_i, D]` | Flattened raw image patches for the vision encoder |
| `image_grid_thw` | `[1, 3]` | The synthetic image's `(T, H, W)` patch grid |

Here `L_i` is the unpadded multimodal token length, `P_i = T * H * W`, and
`D = 3 * temporal_patch_size * patch_size * patch_size`. For a still image,
`T=1`; `temporal_patch_size` is already folded into `D`. Dynamic buckets
make `H`, `W`, `P_i`, and the merged placeholder count vary per sample
while `D` stays constant, so vision embeddings and placeholder positions
remain one-to-one.

When `--pad-packed-seq-alignment` is set, the multimodal packer first keeps
each real sample aligned for static CP/SP, then applies the same CP-local
alignment semantics to the final global packed tail before model-side CP. With
`--pad-packed-seq-by-appending-dummy-seq`, the tail is represented as an
ordinary dummy THD sequence. It adds only token rows, zero loss, and
`padding_mask=true`; it never adds pixels, grid rows, or IMG placeholders.
`--max-seqlen-per-dp-cp-rank` is also required by argument validation and is
the CP-local cap (for example, 128K / CP8 = 16384). The multimodal argument
provider exposes the requested positive
`--pad-packed-seq-by-appending-dummy-seq` compatibility alias; the value is
already true by default, and core's `--no-pad-packed-seq-by-appending-dummy-seq`
can still disable it (which this local packed path rejects when padding is on).

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
