#!/usr/bin/env bash
# See examples/gtp/README.md for hardware, data and environment requirements.
set -euo pipefail
mcore_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$mcore_root"
if [[ -f /etc/shinit_v2 ]]; then source /etc/shinit_v2; fi
export PYTHONPATH="$mcore_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_DEBUG=VERSION
export NCCL_GRAPH_REGISTER=0
export NCCL_NVLS_ENABLE=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_BWD_LAYERNORM_SM_MARGIN=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=32
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
NNODES=${NNODES:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-}}
if (( (NNODES * GPUS_PER_NODE) % 256 != 0 )); then
  echo "World size must be a multiple of 256 for this example's parallel layout." >&2
  exit 2
fi
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
if (( NNODES > 1 )); then
  if (( ! dry_run )); then
    : "${MASTER_ADDR:?Set MASTER_ADDR to the first node address for all nodes}"
    : "${NODE_RANK:?Set NODE_RANK or run one launcher per node with srun}"
  fi
  launcher=(--nnodes "$NNODES" --node_rank "${NODE_RANK:-0}"
    --master_addr "${MASTER_ADDR:-FIRST_NODE}" --master_port "${MASTER_PORT:-29500}")
else
  launcher=(--standalone)
fi
args=(
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model moonshotai/Kimi-K3
  --trust-remote-code
  --vocab-size 163840
  --make-vocab-size-divisible-by 1280
  --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
  --enable-experimental
  --hidden-size 7168
  --ffn-hidden-size 33792
  --num-attention-heads 96
  --kv-channels 64
  --max-position-embeddings 1048576
  --init-method-std 0.02
  --hybrid-layer-pattern K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E+E
  --experimental-attention-variant kda
  --linear-conv-kernel-dim 4
  --linear-key-head-dim 128
  --linear-value-head-dim 128
  --linear-num-key-heads 96
  --linear-num-value-heads 96
  --kda-f-lora-rank 128
  --kda-safe-gate
  --kda-lower-bound -5.0
  --multi-latent-attention
  --q-lora-rank 1536
  --kv-lora-rank 512
  --qk-head-dim 128
  --qk-pos-emb-head-dim 64
  --v-head-dim 128
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
  --num-experts 896
  --moe-ffn-hidden-size 3072
  --moe-latent-size 3584
  --moe-latent-up-projection-rmsnorm
  --moe-shared-expert-intermediate-size 6144
  --moe-router-topk 16
  --moe-router-score-function sigmoid
  --moe-router-dtype fp32
  --moe-router-topk-scaling-factor 1.0
  --moe-router-load-balancing-type quantile_balancing
  --moe-aux-loss-coeff 0.0
  --moe-router-quantile-balancing-estimation-scope global_batch
  --moe-router-qb-num-bins 1000
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --transformer-impl transformer_engine
  --attention-backend fused
  --distributed-timeout-minutes 60
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 64
  --expert-tensor-parallel-size 1
  --context-parallel-size 1
  --distributed-backend nccl
  --moe-token-dispatcher-type flex
  --moe-flex-dispatcher-backend hybridep
  --moe-hybridep-num-sms 32
  --moe-grouped-gemm
  --moe-permute-fusion
  --bf16
  --fp8-format e4m3
  --fp8-recipe mxfp8
  --fp8-param-gather
  --reuse-grad-buf-for-mxfp8-param-ag
  --moe-router-padding-for-quantization
  --seq-length 4096
  --micro-batch-size 1
  --global-batch-size 4096
  --train-iters 1000
  --seed 20260901
  --recompute-granularity selective
  --recompute-modules gdn moe mla_up_proj
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl native
  --optimizer muon
  --use-distributed-optimizer
  --muon-momentum 0.95
  --muon-scale-mode spectral
  --muon-extra-scale-factor 1.0
  --muon-num-ns-steps 5
  --muon-coefficient-type quintic
  --muon-tp-mode auto
  --muon-fp32-matmul-prec highest
  --optimizer-offload-fraction 0.0
  --chunked-optimizer-state-offload
  --optimizer-state-offload-chunk-size-mb 256
  --optimizer-state-offload-fraction 1.0
  --overlap-grad-reduce
  --overlap-param-gather
  --no-create-attention-mask-in-dataloader
  --lr 0.0001
  --min-lr 1e-05
  --lr-warmup-init 1e-05
  --lr-decay-style cosine
  --lr-decay-iters 1000
  --lr-warmup-iters 0
  --weight-decay 0.1
  --clip-grad 1.0
  --manual-gc
  --manual-gc-interval 10
  --eval-iters 0
  --eval-interval 200
  --auto-detect-ckpt-format
  --save-interval 500
  --ckpt-format torch_dist
  --dist-ckpt-strictness raise_all
  --log-throughput
  --log-interval 1
  --logging-level 20
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --log-memory-interval 1
  --log-device-memory-used
  --tensor-parallel-num-weight-shards 64
  --expert-tensor-parallel-num-weight-shards 4
  --gtp-remat-opt-in-modules moe_latent_proj
  --cuda-graph-impl none
  --accumulate-allreduce-grads-in-fp32
)
if [[ -n "${DATA_PATH:-}" ]]; then
  args+=(--data-path "$DATA_PATH" --split "${DATA_SPLIT:-969,30,1}")
else
  args+=(--mock-data)
fi
command=("${PYTHON:-python}" -m torch.distributed.run --nproc_per_node "$GPUS_PER_NODE"
  "${launcher[@]}" "$mcore_root/pretrain_hybrid.py" "${args[@]}" "$@")
if (( dry_run )); then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
