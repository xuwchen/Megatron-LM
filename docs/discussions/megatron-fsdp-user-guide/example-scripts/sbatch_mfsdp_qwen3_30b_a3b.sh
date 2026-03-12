#!/bin/bash

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONWARNINGS=ignore
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_NODEID

# # Configuration: Set these variables before running the script
MEGATRON_PATH=${MEGATRON_PATH:-"your_own_megatron_path"} # Path to Megatron-LM repository
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"your_own_container_image"} # Path to .sqsh or docker image url
OUTPUT_PATH=${OUTPUT_PATH:-"your_own_output_path"} # Path for output logs and checkpoints
DATA_PATH=${DATA_PATH:-"your_own_data_path"}
USE_MEGATRON_FSDP=${USE_MEGATRON_FSDP:-1}
SHARDING_STRATEGY=${SHARDING_STRATEGY:-"optim_grads_params"}
PROFILE=${PROFILE:-0}
WANDB=${WANDB:-1}

TP=${TP:-1}
EP=${EP:-8}
MBS=${MBS:-4}
GBS=${GBS:-256}
COMMENT=${COMMENT:-"alltoall-full-recompute"}

PRETRAIN_ARGS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size ${TP}
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --use-mcore-models
    --sequence-parallel
    --use-flash-attn
    --disable-bias-linear
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-samples 268554688
    --exit-duration-in-mins 230
    --manual-gc
    --manual-gc-interval 5
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --transformer-impl transformer_engine
    --seq-length 4096
    --data-cache-path ${OUTPUT_PATH}/cache
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-30B-A3B
    --data-path ${DATA_PATH}
    --split 99,1,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 6
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
    --make-vocab-size-divisible-by 1187
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --qk-layernorm
    --lr-decay-samples 255126953
    --lr-warmup-samples 162761
    --lr-warmup-init 1.2e-5
    --lr 1.2e-4
    --min-lr 1.2e-5
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95
    --num-experts 128
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-force-load-balancing
    --eval-iters 32
    --eval-interval 100
    --auto-detect-ckpt-format
    --load ${OUTPUT_PATH}/checkpoints
    --save ${OUTPUT_PATH}/checkpoints
    --save-interval 100
    --dist-ckpt-strictness log_all
    --init-method-std 0.02
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --tensorboard-dir ${OUTPUT_PATH}/tensorboard
    --bf16
    --enable-experimental
) 

if [ "${USE_MEGATRON_FSDP}" = 1 ]; then
    unset CUDA_DEVICE_MAX_CONNECTIONS
    PRETRAIN_ARGS=(
        "${PRETRAIN_ARGS[@]}"
        --use-megatron-fsdp
        --data-parallel-sharding-strategy ${SHARDING_STRATEGY}
        --no-gradient-accumulation-fusion
        --use-distributed-optimizer
        --calculate-per-token-loss
        --init-model-with-meta-device
        --ckpt-format fsdp_dtensor
        --grad-reduce-in-bf16
        --fsdp-double-buffer
        --use-nccl-ub
    )
fi

# Profiling command
if [ "${PROFILE}" = 1 ]; then
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx,cublas,cudnn \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        --cuda-memory-usage=true \
        -f true -x true \
        -o ${OUTPUT_PATH}/nsys/Megatron-FSDP-Qwen3-30B-A3B-TP${TP}EP${EP}-MBS${MBS}GBS${GBS}-${COMMENT}"
    PRETRAIN_ARGS=(
        "${PRETRAIN_ARGS[@]}"
        --profile
        --profile-step-start 10
        --profile-step-end 12
        --profile-ranks 0
    )
    echo "PROFILE_CMD="
    echo $PROFILE_CMD
else
    PROFILE_CMD=""
fi

if [ "${WANDB}" = 1 ]; then
    export WANDB_API_KEY=${WANDB_API_KEY:-"your_own_wandb_api_key"}
    PRETRAIN_ARGS=(
        "${PRETRAIN_ARGS[@]}"
        --wandb-project your_own_wandb_project
        --wandb-exp-name Qwen3-30B-A3B-TP${TP}EP${EP}-MBS${MBS}GBS${GBS}-${COMMENT}
    )
fi

TRAINING_CMD="
cd ${MEGATRON_PATH};
git rev-parse HEAD;
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH};
${PROFILE_CMD} python ${MEGATRON_PATH}/pretrain_gpt.py ${PRETRAIN_ARGS[@]}"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Submit SLURM job
# Note: Update SBATCH parameters below according to your cluster configuration
set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=your_own_job_name
#SBATCH --partition=your_own_partition
#SBATCH --nodes=your_own_num_nodes
#SBATCH --ntasks-per-node=your_own_tasks_per_node
#SBATCH --gres=gpu:your_own_gpu_per_node
#SBATCH --time=your_own_time
#SBATCH --account=your_own_account
#SBATCH --exclusive
#SBATCH --dependency=singleton

srun \
    --mpi=pmix -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts="/lustre:/lustre" \
    --container-workdir=${MEGATRON_PATH} \
    bash -x -c "${TRAINING_CMD}" 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log

EOF
set -e
