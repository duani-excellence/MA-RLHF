# 用一个小的 MoE 进行加载调试 verl

set -x

USE_MBRIDGE=True
# ppo_megatron_trainer
NGPUS=2
NBATCHS=2 # 大批量
NMICROBATCHS=2 # 当 microbatch 设置为 2 时, 相当于梯度累计 4 次
NSAMPLES=8

N_TP=1
N_PP=1
N_VPP="null"
N_CP=1
N_EP=$NGPUS
N_DP=1
N_ETP=1

# export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

offload=True
USE_DIST_CKPT=False

export RAY_memory_monitor_refresh_ms=0


rm -rf ./outputs
# max_num_batched_tokens: 2048
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files='/root/Projects/r1/xr1-750/train.parquet' \
    data.val_files='/root/Projects/r1/xr1-750/test.parquet' \
    data.train_batch_size=${NBATCHS} \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path='xiaodongguaAIGC/Qwen3-moe-mini' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${NBATCHS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${NMICROBATCHS} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False  \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${NMICROBATCHS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${NGPUS} \
    actor_rollout_ref.rollout.quantization="fp8" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.n=${NSAMPLES} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${NMICROBATCHS} \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.actor.strategy="megatron" \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${N_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${N_PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${N_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${N_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${N_ETP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${N_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${N_ETP} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${N_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${N_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.ref.megatron.context_parallel_size=${N_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${N_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${N_ETP} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_xr1750' \
    trainer.experiment_name='qwen3_30B_RL_Zero' \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@





    