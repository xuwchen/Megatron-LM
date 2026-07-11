<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software or related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# CUDA Graph

CUDA Graphs reduce kernel-launch overhead by recording GPU operations once and replaying the recording on subsequent iterations. Megatron-LM provides three CUDA graph implementations controlled by `--cuda-graph-impl`.

For implementation background and design details, see NVIDIA's
[Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html).
That article is a useful conceptual reference, but some examples there still use older flags such as
`--enable-cuda-graph` or `--cuda-graph-scope full_iteration`; in this repository, prefer
`--cuda-graph-impl local|transformer_engine|full_iteration` as documented below.

## Overview

CUDA graph behavior is set by three orthogonal flags:

| Flag | Values | Purpose |
|---|---|---|
| `--cuda-graph-impl` | `none` / `local` / `transformer_engine` / `full_iteration` | Which capture backend or strategy to use |
| `--cuda-graph-modules` | `attn` / `mlp` / `moe` / `moe_router` / `moe_preprocess` / `mamba` | Per-layer **training** capture coverage; multi-valued and only meaningful for `local` and `transformer_engine` |
| `--inference-cuda-graph-scope` | `none` / `layer` / `block` | Granularity of CUDA graphs during **inference**; only `local` supports non-`none` values |

Supported combinations:

| `--cuda-graph-impl` | Backend | Training capture | Inference capture |
|---|---|---|---|
| `none` | — | off | off |
| `local` | MCore `CudaGraphManager` | per-layer, controlled by `--cuda-graph-modules` | `layer` (default) or `block`, controlled by `--inference-cuda-graph-scope` |
| `transformer_engine` | TE `make_graphed_callables()` | per-layer, controlled by `--cuda-graph-modules` | not supported (`none` only) |
| `full_iteration` | MCore `FullCudaGraphWrapper` | one graph per training iteration; `--cuda-graph-modules` must be empty | not supported (`none` only) |

---

## CUDA Graph — Local Implementation (`--cuda-graph-impl local`)

Uses MCore's built-in `CudaGraphManager`. During training, this is a per-layer mode:
leaving `--cuda-graph-modules` unset captures the whole Transformer layer, while specifying
modules restricts capture to selected sub-regions. During inference, `local` can instead attach
graphs at either the layer boundary or the enclosing block boundary, as controlled by
`--inference-cuda-graph-scope`.

Operationally, this path is tightly integrated into MCore training and inference:

- graphable modules create and own their `CudaGraphManager` instances automatically
- the existing training schedules drive warmup/capture/replay automatically
- users select the mode through config flags only; there is no separate helper API to
  wire into a custom training loop or a separate need to handle static input buffers

### Usage

```bash
--cuda-graph-impl local
```

### `--cuda-graph-modules` options

| Module | What is captured |
|---|---|
| *(empty / not set)* | Entire Transformer layer (default) |
| `attn` | `TransformerLayer._forward_attention()` |
| `mlp` | `TransformerLayer._forward_mlp()` for dense layers |
| `moe` | `TransformerLayer._forward_mlp()` for MoE layers (drop-and-pad only) |
| `moe_router` | MoE router + shared experts (if not EP-comm-overlapped) |
| `moe_preprocess` | `MoELayer.preprocess()` — must be paired with `moe_router` |
| `mamba` | Mamba SSM layer |

**Example — MoE model, capture attention and router:**
```bash
--cuda-graph-impl local \
# Optionally restrict captured modules (default: capture whole layer, but not working for MoE dynamic shapes)
--cuda-graph-modules attn moe_router moe_preprocess
```

---

## CUDA Graph — Transformer Engine Implementation (`--cuda-graph-impl transformer_engine`)

Uses Transformer Engine's `make_graphed_callables()` path. In Megatron-LM's CLI, this has the
same training granularity as `local`: leaving `--cuda-graph-modules` unset captures the whole
Transformer layer, while specifying modules restricts capture to selected sub-regions. The main difference from
`local` is the backend implementation and feature compatibility. Unlike `local`, this path does
not support inference CUDA graphs.

Compared to `local`, this path exposes a more general and self-contained API via TE's
`make_graphed_callables()`, giving users greater flexibility and control over how CUDA graphs are
wired into custom training loops. The trade-off is that it requires more manual setup:

- the training loop must instantiate `TECudaGraphHelper`
- the training loop must call helper methods such as `create_cudagraphs()` and
  `cuda_graph_set_manual_hooks()` at the correct points

Megatron-LM's stock training loop already wires these calls in `megatron/training/training.py`,
but custom training scripts must do the same work themselves.

### Usage

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

The same training `--cuda-graph-modules` options apply as for `local`, and the default is likewise
whole-layer training capture when the flag is omitted.

### Megatron-FSDP planned double buffering

With Megatron-FSDP, the Transformer Engine backend uses a planned, decoder-only double buffer to
keep the temporary parameter and main-gradient addresses referenced by CUDA graphs stable. An
eager warmup records decoder FSDP-unit lifetimes and assigns overlapping units to two static banks.
All buckets in one FSDP unit use the same bank color. Parameter all-gather and gradient
reduce-scatter scheduling are limited to two decoder units so that a third live unit cannot
overwrite either bank. A decoder unit cannot reclaim its main-gradient bank until any earlier
reduce-scatter reading the same unit has completed. Before a fused weight-gradient CUDA graph
writes `main_grad`, Megatron-FSDP claims and validates that unit's planned bank.

Within each parameter-and-gradient buffer (one model chunk), the two banks are shared by
all language-decoder layers. For each storage kind and dtype, each bank's arena is sized to the
maximum aggregate bucket size of any decoder FSDP unit. Vision encoder buckets are not included in
this plan and continue to use the dynamic allocator. Vision encoder TE
capture is unsupported in this mode, so Vision computation remains eager. Under Megatron-FSDP,
mHC-wrapped layers remain eager.

This mode currently requires:

- `--data-parallel-sharding-strategy optim_grads_params`
- `--fsdp-double-buffer`
- `--overlap-param-gather` and `--overlap-grad-reduce`
- `--cuda-graph-warmup-steps >= 1`
- `--fsdp-db-use-persist-buf-on-alloc-fail` to remain disabled (the default)
- fine-grained parameter gather to remain disabled, including explicit
  `--megatron-fsdp-enable-fine-grained-param-gather` and the `mxfp8` recipe combined with
  `--fp8-param-gather`; `--overlap-moe-expert-parallel-comm` is also unsupported because it
  enables the same fine-grained FSDP hooks internally
- `--use-nccl-ub` to remain disabled; planned banks are not yet compatible with NCCL user-buffer
  registration

`start_param_sync(force_sync=True)` is unsupported because a whole-model unshard cannot remain
resident in a frozen two-bank plan; the normal overlapped per-unit hooks must drive parameter sync.

`MEGATRON_CG_SKIP_BUFFER_ADDRESS_CHECK=1` disables the replay-time parameter-address check for
diagnostics only. Other values, including `0` and `false`, leave the check enabled. Bypassing it
can turn an allocator error into silent corruption and is not supported for production runs.

The planned bank lifetime topology observed during warmup must not change afterward; dynamic
microbatch shapes, FSDP-unit topology, or communication schedules are not supported in this mode.
A capture failure is fatal for the run and is not retryable. Cross-rank capture-failure consensus is
deferred to follow-up work.

For MoE decoders, the intended partial capture scopes are `attn`, `moe_router`, and
`moe_preprocess`. Gated Delta Net (GDN) blocks are captured under the `attn` scope only with the
explicit opt-in environment variable `MEGATRON_GDN_TE_CUDA_GRAPH=1`:

```bash
MEGATRON_GDN_TE_CUDA_GRAPH=1 python pretrain_multimodal.py \
  --use-megatron-fsdp \
  --data-parallel-sharding-strategy optim_grads_params \
  --cuda-graph-impl transformer_engine \
  --cuda-graph-modules attn moe_router moe_preprocess \
  --cuda-graph-warmup-steps 3 \
  --fsdp-double-buffer
```

---

## Full-Iteration Training CUDA Graph (`--cuda-graph-impl full_iteration`)

Captures the entire training iteration (excluding optimizer) as a single CUDA graph. The same
wrapper is also used for training-loop validation/eval in forward-only mode. This provides the
largest training/validation latency reduction.

This implementation does not create inference CUDA graphs. For inference, use
`--cuda-graph-impl local --inference-cuda-graph-scope layer|block`.

### Requirements

- `--no-check-for-nan-in-loss-and-grad` is required: NaN checks involve CPU-GPU synchronization
  which cannot run inside a CUDA graph.
- `--cuda-graph-modules` must be omitted (or left empty): per-module selection has no meaning
  when the entire iteration is captured as a single graph.

### Example

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

---

## Common Configuration Examples

### Dense Model Training

All three implementations work for dense models:

```bash
# Per-layer (local)
--cuda-graph-impl local
# equivalent: --cuda-graph-impl local --cuda-graph-modules attn mlp

# Per-layer (TE)
--cuda-graph-impl transformer_engine
# equivalent: --cuda-graph-impl transformer_engine --cuda-graph-modules attn mlp

# Full-iteration
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

### MoE Model Training

MoE expert dispatch involves dynamic shapes and cannot be captured. `--cuda-graph-modules` is used
to capture only the static parts (attention, router, preprocess) while leaving expert compute in
eager mode. Example using `transformer_engine` (`local` works the same way):

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

With [paged stash](paged_stash.md), expert dispatch shapes become static (pre-sized via
`--moe-expert-rank-capacity-factor`), which allows full-iteration CUDA graphs to be used on MoE
models as well:

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad \
--moe-flex-dispatcher-backend hybridep \
--use-transformer-engine-op-fuser \
--moe-expert-rank-capacity-factor <float> \
--moe-paged-stash
```

---

## Additional Notes

- `--cuda-graph-warmup-steps` (default: 3) controls how many warmup steps run before CUDA graph
  capture. Setting it to 0 is not recommended: some operations rely on the first few iterations
  for lazy initialization or autotuning, and capturing too early may produce incorrect or
  suboptimal graphs.
- Inference CUDA graphs (serving or RL rollout) currently require
  `--cuda-graph-impl local`. Use `--inference-cuda-graph-scope layer|block` with
  `local`; all other implementations must set `--inference-cuda-graph-scope none`,
  meaning inference runs in eager mode.
- Background reference: [Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html),
  which also covers PyTorch CUDA Graph best practices and lessons learned.

---

## Migration Guide

Legacy configurations (including `--enable-cuda-graph`, `--external-cuda-graph`, the renamed
`--cuda-graph-scope` flag (now `--cuda-graph-modules`), and deprecated module values such as
`full_iteration` and `full_iteration_inference`) are still accepted and automatically migrated
at runtime, but we encourage updating your configs to the new forms:

| Old command | New command |
|---|---|
| `--enable-cuda-graph` | `--cuda-graph-impl local` |
| `--external-cuda-graph` | `--cuda-graph-impl transformer_engine` |
| `--cuda-graph-scope <modules>` | `--cuda-graph-modules <modules>` |
| `--cuda-graph-impl local --cuda-graph-scope full_iteration` | `--cuda-graph-impl full_iteration` |
| `--cuda-graph-impl local --cuda-graph-scope full_iteration_inference` | `--cuda-graph-impl local --inference-cuda-graph-scope block` |
| `--cuda-graph-impl local --cuda-graph-scope attn moe_router moe_preprocess full_iteration_inference` | `--cuda-graph-impl local --cuda-graph-modules attn moe_router moe_preprocess --inference-cuda-graph-scope block` |
