#!/bin/bash

# cp ./run_fully_async_4B_4gpus.sh ~/Projects/slime/examples/fully_async/
# cd ~/Projects/slime
# bash ./run_fully_async_4B_4gpus.sh

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex


# 该处设定运行环境
# MY_GPUS_PER_NODE = MEGATRON_GPUS+SGLANG_GPUS
# 默认 GPU 数量等同 TP 并行数
MY_NODE=1
MY_GPUS_PER_NODE=4 
MEGATRON_GPUS=2
SGLANG_GPUS=2


# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# 根据实际模型修改配置, 不然 Megatron 会加载不了模型
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh" 

# 注意路径
CKPT_ARGS=(
   --hf-checkpoint /data/Qwen3-4B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load  /data/Qwen3-4B-dist
   --load /data/Qwen3-4B_slime/
   --save /data/Qwen3-4B_slime/
   --save-interval 100
)

# 注意路径
PROMPT_SET=/data/dapo-math-17k/dapo-math-17k.jsonl


ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout 3000
   # --rollout-batch-size 32
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   # --global-batch-size 64
   --global-batch-size 64
   --balance-data
)

# Megatron: TP=2
PERF_ARGS=(
   --tensor-model-parallel-size ${MEGATRON_GPUS}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 5000
)

# Sglang: TP=2
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)


GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # offload
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
)


MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)


WANDB_ARGS=(
   --use-wandb
   --wandb-project fully-async-4B-4gpu
   --wandb-group slime-fully-async
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH=$PYTHONPATH:/root/Projects/Megatron-LM

# 总 GPUs = 4
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${MY_GPUS_PER_NODE} --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Projects/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# node=1
# Megatron: GPUs=2, `--actor-num-gpus-per-node 2` -> TP=2
# Sglang: GPUs=2, `--rollout-num-gpus 2` -> TP=2
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${MEGATRON_GPUS} \
   --rollout-num-gpus ${SGLANG_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} 
   