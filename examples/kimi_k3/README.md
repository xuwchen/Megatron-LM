# Kimi-K3 + GTP quick start

Run from this Megatron checkout inside a GPU container with four H100 80GB GPUs:

```bash
bash examples/kimi_k3/run_slim_gtp4.sh
```

This is the **29-layer slim text proxy**, not the full multimodal model. It uses
mock tokens, BF16, SL4096, MBS1/GBS4, GTP4/EP4/EGTP1, HybridEP, deterministic
mode, FP32 gradient accumulation, and duplicated/highest Muon. It retains
selective KDA/MoE recomputation and chunked optimizer-state CPU offload. The
router learns normally. It starts from random weights and needs no checkpoint
or correctness observer. The default stops after five updates while retaining
a 1000-update learning-rate/weight-decay schedule.

```bash
# Print the complete command without starting training.
bash examples/kimi_k3/run_slim_gtp4.sh --dry-run
# Run 50 updates with the same scheduler horizon.
bash examples/kimi_k3/run_slim_gtp4.sh --exit-interval 50
# Save model and optimizer state at update 50 (use a new output directory).
bash examples/kimi_k3/run_slim_gtp4.sh --exit-interval 50 \
  --save /path/to/new/checkpoints --save-interval 50
```

Additional arguments are appended to Megatron's command. Topology is fixed to a
single four-GPU node; changing only EP or GTP does not produce a validated recipe.
The script must execute on an allocated GPU node, not a cluster login node.

## Environment

The measured CW environment is NGC PyTorch 26.03 (CUDA 13.2), TransformerEngine
2.20.0.dev0 built from `0a2ebe942b9f6a66aa0292502966e73114a7b912` with GTP and
SiTUGLU, FLA 0.5.2, emerging-optimizers 0.3.0, transformers 5.8.1, and working
HybridEP/deep_ep binaries. Stock TE packages without GTP/SiTUGLU are insufficient.
The tokenizer is `moonshotai/Kimi-K3`; make it available through the HF cache or
normal HF access. This script does not download model weights.

On the existing CW container setup, select the prepared dependency overlays:

```bash
export KIMI_DEPS_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/users/xuwenc/agentic-mcore-dev/runtime
bash examples/kimi_k3/run_slim_gtp4.sh
```

`KIMI_DEPS_ROOT` is optional when all dependencies are already installed. The
script sources `/etc/shinit_v2` if present for NGC driver compatibility. For
cluster allocation/container setup, the companion toolkit backup contains
`scripts/run_kimi_k3_gtp_cw.sh` and the `h100/bf16_gtp4_start` recipe.

## Backup and validation scope

Branch: `backup/gtp-kimi-k3-20261013`. The parent production code is
`0a2eddec02ffade3ee6ce50a184a3153c0dbe364`; the original
`fix/kimi-k3-hf-order-20261011` branch remains at that commit. This backup adds
only the startup script and documentation to Megatron.

The previous four 50-update comparisons used this production source and the
same numerical settings with a shared model/optimizer checkpoint. Maximum
native LM-loss absolute difference was 0.00482711196. Complete gradients,
weights and optimizer states were not identical; broader real-data convergence
has not been established. A fresh five-update smoke run is a functionality
check, not a replay of that comparison. HF forward checking is a separate
workflow in [HF_PARITY.md](HF_PARITY.md).
