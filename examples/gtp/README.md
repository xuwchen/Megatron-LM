# GTP training launch examples

All commands below run from this Megatron-LM checkout inside a provisioned GPU
container. They use normal training entrypoints, not correctness observers.
Each script supports `--dry-run` and appends additional Megatron arguments.
The scripts and their source are in `backup/gtp-kimi-k3-20261013`.

| Script | Model scope | Default hardware/layout | Precision / optimizer | Data |
|---|---|---|---|---|
| `examples/kimi_k3/run_slim_gtp4.sh` | Kimi-K3 29-layer slim text proxy | 1 x 4 H100; GTP4, EP4, EGTP1, TP1/PP1/CP1 | BF16; duplicated/highest Muon | Mock tokens |
| `examples/gtp/run_qwen35_vl_35b.sh` | Full Qwen3.5-VL 35B-A3B: 40 language blocks, 27 vision layers, MTP1 | 4 x 4 GB200; GTP4, EP8, EGTP2, TP1/PP1/CP1 | BF16; distributed Adam; deterministic; full decoder recompute | CORD-v2 image/text |

GTP and EGTP are weight-rematerialization degrees. The actual CLI flags are
`--tensor-parallel-num-weight-shards` and
`--expert-tensor-parallel-num-weight-shards` (including TP/ETP respectively).
The Kimi example is a text-only slim proxy. The Qwen example includes the full
35B-A3B language and vision architecture; its exact configuration has not been
GPU-tested again on this backup branch. This task exercises the slim entrypoint.

## 1. Kimi-K3 slim on four GPUs

```bash
# Optional: use the existing CW H100 dependency overlays inside the container.
export KIMI_DEPS_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/users/xuwenc/agentic-mcore-dev/runtime
bash examples/kimi_k3/run_slim_gtp4.sh
# Longer run; the learning-rate horizon stays at 1000 updates.
bash examples/kimi_k3/run_slim_gtp4.sh --exit-interval 50
```

For CW scheduling from the companion toolkit backup:

```bash
bash scripts/run_kimi_k3_gtp_cw.sh dry-run
bash scripts/run_kimi_k3_gtp_cw.sh run
```

## 2. Qwen3.5-VL 35B-A3B on 16 GPUs

```bash
export NNODES=4 GPUS_PER_NODE=4
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29500
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  bash examples/gtp/run_qwen35_vl_35b.sh \
  --load /shared/checkpoints/qwen35-vl-35b-mcore --finetune \
  --save /shared/checkpoints/qwen35-vl-35b-gtp --save-interval 100
```

The load path must be a converted Megatron checkpoint with the matching MTP and
vision architecture. `--finetune` resets optimizer/counters; omit it to resume a
compatible training checkpoint. Omit both load options to train from random
initialization. This is the GTP/DDP path, with no Megatron FSDP flags.
The script uses the repository's `cord_v2` provider, which loads
`naver-clova-ix/cord-v2`, and the shared Qwen3.5 tokenizer/processor identifier
`Qwen/Qwen3.5-397B-A17B`; it does not load 397B model weights. Populate the HF
cache or allow dataset/processor downloads. To use a different dataset, implement
or select an existing `multimodal_dev` dataset provider; a text-only `--data-path`
does not replace the CORD provider.

## Environment and rendering

Kimi requires TE with both GTP and SiTUGLU, FLA 0.5.2, emerging-optimizers 0.3.0,
and working HybridEP. The prior CW setup uses PyTorch 26.03 + TE built from
`0a2ebe942b9f6a66aa0292502966e73114a7b912`.

The prior Qwen environment was PyTorch 26.07/CUDA13.3, TE
`2.20.0.dev0+2efa03ea`, compatible FLA/causal-conv1d, HybridEP, transformers and
datasets. Its OCI image was
`/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/shiqingf/images/nt6_hybrid/nvidia+pytorch+26.07-py3+deps+v10.sqsh`.
Use a provisioned environment with these capabilities; an unmodified stock TE
installation may not support this branch's GTP interfaces.

```bash
bash examples/kimi_k3/run_slim_gtp4.sh --dry-run
bash examples/gtp/run_qwen35_vl_35b.sh --dry-run
```

The examples leave router learning enabled and do not register W&B runs. Save,
load, logging and dataset paths are supplied explicitly by the caller. These
launch recipes do not by themselves establish numerical parity or convergence.
