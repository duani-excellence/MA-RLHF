# FrameWork: verl + sglang + megatron
# ENV: 3090(24GB)x2 ~2h / 1 epochs
# TP: 2
# Data: xr1-750
# Result: 57.23%(test set)

set -x

USE_MBRIDGE=FALSE
# ppo_megatron_trainer
NGPUS=4
NBATCHS=8 # 大批量
NMICROBATCHS=2 # 当 microbatch 设置为 2 时, 相当于梯度累计 4 次
NSAMPLES=8

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files='./xr1-750/train.parquet' \
    data.val_files='./xr1-750/test.parquet' \
    data.train_batch_size=$NBATCHS \
    data.val_batch_size=$NBATCHS \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="/data/Qwen3-0.6B-Base" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$NMICROBATCHS \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$NMICROBATCHS \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$NMICROBATCHS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$NGPUS \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.n=$NSAMPLES \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$NMICROBATCHS \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.actor.strategy="megatron" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$NGPUS \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_xr1750' \
    trainer.experiment_name='qwen3_0dot6B_RL_Zero' \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@