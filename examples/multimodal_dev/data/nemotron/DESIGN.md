# Nemotron MDP data contract

## Boundaries

This package owns source preparation and individual records. The model registry
lazily resolves `nemotron` to `training_provider.py`. The standard entrypoint,
sampler, greedy wrapper, THD collator and MDP scheduler retain their existing
ownership. There is no second packing implementation or custom training loop.

The converter joins JSONL messages with ordered media references and writes
paired `<subset>__<id>.json` and `.jpgs` members, the latter a trusted pickle list
of original bytes. It assigns a deterministic hash-based 95/5 train/validation
split. Offline Energon indexes retain Bridge ChatML WebDataset compatibility;
training instead uses a map-style offset index.

## Admission

Every self-contained subset follows the same `prepare_training.py` stages:
verify pinned source sizes/SHA-256, convert or validate reused conversion, index,
assess all source lengths, select whole records, and validate actual Dataset
outputs. SEC and CLEVR have no family-specific bypass.

The tar index verifies split counts, paired components and unique keys without
decoding images or tokens. The census processes every key, derives image grids
at the pinned resolution, and checks deterministic samples against full pixel
processing. Selection requires exact census coverage and matching source SHA-256;
a length-file hash binds the decision to the assessed lengths.

Each split keeps source order and only records with `0 < length <= T`. Selection
stores source provenance, processor ID/revision/resolution, budget, split keys
and counts. Actual-Dataset validation includes the longest two admitted records
per split. New selections are published through a temporary file and rename;
identical selections remain byte-stable and conflicting existing ones fail.
Preparation expects a single writer per prepared directory.

Runtime requires a selection JSON and rechecks budget, processor, index
provenance, unique keys and split membership. It does not rehash each tar on each
worker or rerun the census. Keep prepared artifacts immutable. Validation
receipts are audit outputs; training consumes the selection and index themselves.

## Records and sampling

`NemotronDataset` caches a processor and up to eight tar handles per worker.
Spawn serialization clears these resources. It returns `input_ids`, shifted
`labels` and `loss_mask`, BF16 `pixel_values`, and `image_grid_thw` after reading
the complete conversation and ordered images.

Assistant spans come from the rendered ChatML token stream because Qwen's
template can normalize earlier reasoning blocks. Only assistant prediction
targets contribute; special tokens are excluded. Empty targets, budget violations,
vocabulary overflow and image-placeholder/grid mismatches raise with the source
key. The adapter does not truncate, drop images or substitute mock content.

Native samplers own data-parallel partitioning and cyclic ordering. A virtual
split length covers the requested pack slots, greedy real-record cap `K-1` and
lookahead. Repetition stays within that split. Real data has no separate test
split. Native greedy loading owns commit accounting and collation; exact resume
is not supplied by this adapter.

## Portable experiment

`convergence_32k.json` contains shared model, training, data-shape and runtime
environment controls. `run_convergence.py` builds a shell-free torchrun command,
adding dataset/selection and W&B/output identity. The source recipe's shared
arguments are preserved, including PP2/EP2, 32K static THD and 8192-row encoder
chunking plus whole replay. Output directories must be fresh. Dry-run prints
only the command and explicit runtime environment, without reading credentials.

CPU tests check admission and runtime contracts, matched experiment controls and
complete scalar histories. The new portable entrypoint has not had its own GPU
rerun; forward/backward validation remains a smoke/full-training step.
