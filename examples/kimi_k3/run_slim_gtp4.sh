#!/usr/bin/env bash
# Fresh 29-layer slim text proxy. Requires four GPUs and the dependencies in README.md.
set -euo pipefail
mcore_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$mcore_root"
# Inside NGC/Pyxis, activate the image's CUDA driver compatibility setup.
if [[ -f /etc/shinit_v2 ]]; then source /etc/shinit_v2; fi
if [[ -n "${KIMI_DEPS_ROOT:-}" ]]; then
  for dependency in "$KIMI_DEPS_ROOT/gtp_kimi_k3/te_0a2ebe_h100" \
      "$KIMI_DEPS_ROOT/nvrx/0.6.0-h100-torch2603" \
      "$KIMI_DEPS_ROOT/gtp_kimi_k3/deps_h100"; do
    [[ -d "$dependency" ]] || { echo "Missing dependency overlay: $dependency" >&2; exit 1; }
  done
  export PYTHONPATH="$KIMI_DEPS_ROOT/gtp_kimi_k3/te_0a2ebe_h100:$KIMI_DEPS_ROOT/nvrx/0.6.0-h100-torch2603:$KIMI_DEPS_ROOT/gtp_kimi_k3/deps_h100${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$mcore_root${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_ALGO=Ring
export NCCL_GRAPH_REGISTER=0
export NCCL_NVLS_ENABLE=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=32
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export PYTHONHASHSEED=20260903
export PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=0

args=(
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model moonshotai/Kimi-K3
  --trust-remote-code
  --vocab-size 163840
  --make-vocab-size-divisible-by 1280
  --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
  --enable-experimental
  --num-layers 58
  --hybrid-layer-pattern K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKE
  --experimental-attention-variant kda
  --hidden-size 1024
  --ffn-hidden-size 4096
  --num-attention-heads 16
  --kv-channels 64
  --max-position-embeddings 1048576
  --init-method-std 0.02
  --linear-conv-kernel-dim 4
  --linear-key-head-dim 64
  --linear-value-head-dim 64
  --linear-num-key-heads 32
  --linear-num-value-heads 32
  --kda-f-lora-rank 64
  --kda-safe-gate
  --kda-lower-bound -5.0
  --multi-latent-attention
  --q-lora-rank 256
  --kv-lora-rank 128
  --qk-head-dim 64
  --qk-pos-emb-head-dim 32
  --v-head-dim 64
  --qk-layernorm
  --attention-latent-norm-epsilon 1e-06
  --attention-output-gate
  --gated-attention-proj-granularity elementwise
  --no-rope-freq 1
  --position-embedding-type none
  --enable-attention-residuals
  --attn-res-block-layers 24
  --normalization RMSNorm
  --norm-epsilon 1e-05
  --situ-glu
  --situ-glu-beta1 4.0
  --situ-glu-beta2 25.0
  --disable-bias-linear
  --untie-embeddings-and-output-weights
  --num-experts 32
  --moe-ffn-hidden-size 512
  --moe-latent-size 512
  --moe-latent-up-projection-rmsnorm
  --moe-shared-expert-intermediate-size 512
  --moe-router-topk 2
  --moe-router-score-function sigmoid
  --moe-router-dtype fp32
  --moe-router-topk-scaling-factor 1.0
  --moe-router-load-balancing-type quantile_balancing
  --moe-aux-loss-coeff 0.0
  --moe-router-quantile-balancing-estimation-scope global_batch
  --moe-router-qb-num-bins 1000
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --mock-data
  --seq-length 4096
  --transformer-impl transformer_engine
  --attention-backend fused
  --distributed-timeout-minutes 60
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 4
  --expert-tensor-parallel-size 1
  --context-parallel-size 1
  --distributed-backend nccl
  --moe-token-dispatcher-type flex
  --moe-grouped-gemm
  --moe-permute-fusion
  --bf16
  --micro-batch-size 1
  --global-batch-size 4
  --train-iters 1000
  --seed 20260903
  --lr 0.0001
  --min-lr 1e-05
  --lr-warmup-init 1e-05
  --lr-decay-style cosine
  --lr-decay-iters 1000
  --lr-warmup-iters 0
  --weight-decay 0.1
  --clip-grad 1.0
  --optimizer muon
  --use-distributed-optimizer
  --muon-momentum 0.95
  --muon-scale-mode spectral
  --muon-extra-scale-factor 1.0
  --muon-num-ns-steps 5
  --muon-coefficient-type quintic
  --muon-tp-mode duplicated
  --muon-fp32-matmul-prec highest
  --optimizer-offload-fraction 0.0
  --chunked-optimizer-state-offload
  --optimizer-state-offload-chunk-size-mb 256
  --optimizer-state-offload-fraction 1.0
  --recompute-granularity selective
  --recompute-modules gdn moe
  --overlap-grad-reduce
  --overlap-param-gather
  --no-create-attention-mask-in-dataloader
  --cross-entropy-fusion-impl native
  --manual-gc
  --manual-gc-interval 10
  --eval-interval 1000
  --eval-iters 0
  --ckpt-format torch_dist
  --log-interval 1
  --log-throughput
  --deterministic-mode
  --use-cpu-initialization
  --log-num-zeros-in-grad
  --log-params-norm
  --log-memory-interval 5
  --log-memory-to-tensorboard
  --tensorboard-log-interval 1
  --dist-ckpt-strictness raise_all
  --exit-interval 5
  --moe-flex-dispatcher-backend hybridep
  --accumulate-allreduce-grads-in-fp32
  --cuda-graph-impl none
  --calculate-per-token-loss
  --tensor-parallel-num-weight-shards 4
  --expert-tensor-parallel-num-weight-shards 1
  --num-workers 0
  --log-device-memory-used
)
command=("${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node=4
  "$mcore_root/pretrain_hybrid.py" "${args[@]}")
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' "${command[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${command[@]}" "$@"
