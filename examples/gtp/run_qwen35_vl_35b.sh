#!/usr/bin/env bash
# See examples/gtp/README.md for hardware, data and environment requirements.
set -euo pipefail
mcore_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$mcore_root"
if [[ -f /etc/shinit_v2 ]]; then source /etc/shinit_v2; fi
export PYTHONPATH="$mcore_root${PYTHONPATH:+:$PYTHONPATH}"
export CAUSAL_CONV1D_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MAMBA_DETERMINISTIC=1
export NCCL_ALGO=Ring
export NCCL_IB_SL=1
export NCCL_NVLS_ENABLE=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=32
export NVTE_FUSED_ATTN=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-}}
if (( (NNODES * GPUS_PER_NODE) % 16 != 0 )); then
  echo "World size must be a multiple of 16 for this example's parallel layout." >&2
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
  --micro-batch-size 1
  --global-batch-size 128
  --train-iters 1000
  --adam-beta1 0.9
  --adam-beta2 0.95
  --lr 0.0001
  --min-lr 1e-05
  --lr-decay-style cosine
  --lr-warmup-iters 50
  --lr-decay-iters 1000
  --weight-decay 0.1
  --clip-grad 1.0
  --bf16
  --use-mcore-models
  --transformer-impl transformer_engine
  --cross-entropy-fusion-impl native
  --enable-experimental
  --manual-gc
  --manual-gc-interval 50
  --sft
  --use-flash-attn
  --calculate-per-token-loss
  --mtp-num-layers 1
  --mtp-loss-scaling-factor 0.1
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 8
  --context-parallel-size 1
  --cp-comm-type a2a
  --expert-tensor-parallel-size 1
  --use-distributed-optimizer
  --log-interval 1
  --eval-interval 1000
  --eval-iters 0
  --log-throughput
  --log-timers-to-tensorboard
  --log-params-norm
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model Qwen/Qwen3.5-397B-A17B
  --model-arch qwen35_vl
  --model-variant 35b_a3b
  --dataset-provider cord_v2
  --hf-processor-path Qwen/Qwen3.5-397B-A17B
  --use-vanilla-collate-fn
  --image-token-id 248056
  --image-size 224
  --total-seq-length 4096
  --image-seq-length 256
  --vision-num-layers 27
  --hidden-size 2048
  --ffn-hidden-size 4096
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 2
  --kv-channels 256
  --max-position-embeddings 262144
  --seq-length 4096
  --normalization RMSNorm
  --apply-layernorm-1p
  --norm-epsilon 1e-06
  --swiglu
  --disable-bias-linear
  --position-embedding-type mrope
  --mrope-section 11 11 10
  --rotary-percent 0.25
  --rotary-base 10000000
  --rotary-seq-len-interpolation-factor 1
  --qk-layernorm
  --attention-output-gate
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --experimental-attention-variant gated_delta_net
  --linear-conv-kernel-dim 4
  --linear-key-head-dim 128
  --linear-value-head-dim 128
  --linear-num-key-heads 16
  --linear-num-value-heads 32
  --make-vocab-size-divisible-by 485
  --untie-embeddings-and-output-weights
  --hybrid-layer-pattern 'GEGEGE*EGEGEGE*EGEGEGE*EGEGEGE*EGEGEGE*EGEGEGE*EGEGEGE*EGEGEGE*EGEGEGE*EGEGEGE*E/*E'
  --num-experts 256
  --moe-ffn-hidden-size 512
  --moe-shared-expert-intermediate-size 512
  --moe-shared-expert-gate
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk 8
  --moe-grouped-gemm
  --moe-aux-loss-coeff 1e-3
  --moe-token-dispatcher-type flex
  --moe-router-dtype fp32
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --tensor-parallel-num-weight-shards 4
  --expert-tensor-parallel-num-weight-shards 2
  --moe-flex-dispatcher-backend hybridep
  --ckpt-format torch_dist
  --cuda-graph-impl none
  --accumulate-allreduce-grads-in-fp32
  --deterministic-mode
  --moe-hybridep-pad-variable-tokens
  --gtp-remat-pad-for-alignment 32
  --overlap-grad-reduce
  --dist-ckpt-strictness raise_all
  --seed 1234
  --log-memory-interval 5
  --log-device-memory-used
)
command=("${PYTHON:-python}" -m torch.distributed.run --nproc_per_node "$GPUS_PER_NODE"
  "${launcher[@]}" "$mcore_root/examples/multimodal_dev/pretrain_multimodal.py" "${args[@]}" "$@")
if (( dry_run )); then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
