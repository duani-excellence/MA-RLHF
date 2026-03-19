# from verl/experimental/fully_async_policy/shell/dapo_7b_async_retool.sh

set -x

export VLLM_USE_V1=1

# ================= data/model/tool =================
DATA_ROOT=${DATA_ROOT:'/data'}

dapo_math_17k=$DATA_ROOT/DAPO-Math-17k
aime_2025=$DATA_ROOT/aime_2025
model_path='Qwen/Qwen3-0.6B-Base'

train_files="['$dapo_math_17k']"
test_files="['$aime_2025']"

# tool
tool_config_path=./sandbox_fusion_tool_config.yaml
retool_path=./retool.py

# wandb / tensorboard
project_name=retool
experiment_name=Qwen3-0.6B_dapo_async_tool
default_local_dir=$DATA_ROOT/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
# kl_coef=0.0
# use_kl_loss=False
# kl_loss_coef=0.0

# clip_ratio_low=0.2
# clip_ratio_high=0.28

max_turns=4
max_prompt_length=256
max_response_length=4096
actor_lr=1e-6

# ================= perfomance =================
NGPUS=4
infer_tp=2 # vllm
# train_sp=2 # train
train_tp=2 # train for megatron
fsdp_size=2 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

# ================= async policy =================
rollout_name="vllm"
rollout_mode="async"

NNODES=1
n_gpus_rollout=2
n_gpus_training=2

train_batch_size=0
ppo_mini_batch_size=4
gen_prompt_bsz=1
n_resp_per_prompt=4
n_resp_per_prompt_val=8
total_rollout_steps=512
test_freq=10

# 此参数控制异步训练中，rollout的结果被认为过时（stale）的阈值。具体来说，如果一个rollout的结果在被训练器（trainer）使用时已经过了这个时间阈值，那么这个结果就会被丢弃，不用于更新模型。这有助于确保模型更新时使用的rollout结果是相对新鲜的，从而提高训练的稳定性和效果。
staleness_threshold=0.5 

trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$retool_path \
    reward.custom_reward_function.path=$retool_path \
    data.custom_cls.name=CustomRLHFDataset \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=True \
    trainer.log_val_generations=20 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$default_local_dir \
    data.gen_batch_size=${gen_prompt_bsz} \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$n_gpus_training \
    rollout.nnodes=$NNODES \
    rollout.n_gpus_per_node=$n_gpus_rollout \
    rollout.total_rollout_steps=$total_rollout_steps \
    rollout.total_epochs=10 \
    rollout.test_freq=$test_freq \
    async_training.staleness_threshold=$staleness_threshold \
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step \
    async_training.require_batches=$require_batches \
    async_training.partial_rollout=$partial_rollout




    # actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \

    # critic.strategy=fsdp2 \