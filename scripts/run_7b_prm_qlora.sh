# 完整运行
# Llama 的模型需要提前在huggingface申请license, Llama3.2和Llama3.3 没有8B的版本
# base_model_path='meta-llama/Meta-Llama-3-8B' # base
base_model_path='meta-llama/Llama-3.1-8B'
# base_model_path='unsloth/Llama-3.2-3B'
# base_model_path='meta-llama/Llama-3.2-3B'    # Llama
# base_model_path='xiaodongguaAIGC/xdg-llama-3-8B'
# base_model_path='xiaodongguaAIGC/llama-3-debug'
deepspeed_config_name=./config/ds_stage3.json
output_path='./output'

# model_pretrained_full_path=${base_model_path}
model_pretrained_full_path=${base_model_path}
model_sft_step_lora_path=${output_path}'/step_sft_lora'
model_sft_step_full_path=${output_path}'/step_sft_full'
model_prm_lora_path=${output_path}'/prm_lora'
model_prm_full_path=${output_path}'/prm_full'


echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'


# # stage: sft
sft_dataset_name='xiaodongguaAIGC/step_sft' # GSM8k_step_sft
deepspeed ./ma-rlhf/sft_step.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=1024 \
	--output_name=${model_sft_step_lora_path} \
	--use_QLora=True \
	--batch_size=4 \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=2 \
	--gradient_accumulation_steps=8 \
	--learning_rate=2e-5

#  merge SFT with lora
#  由于lm_head lora, 合并时间较长。
 python ./ma-rlhf/merge_adapter.py \
 	--base_model_name=${model_pretrained_full_path} \
 	--model_name=${model_sft_step_lora_path} \
 	--merged_model_name=${model_sft_step_full_path}

bash ./scripts/run_step_generation_examples.sh ${model_sft_step_lora_path} 2048

# # # # stage: prm
prm_dataset_name='xiaodongguaAIGC/step_prm'
# deepspeed ./ma-rlhf/sft_step.py \
deepspeed ./ma-rlhf/process_reward_model.py \
	--dataset_name=${prm_dataset_name} \
	--model_name=${model_sft_step_full_path} \
	--seq_length=512 \
	--output_name=${model_prm_lora_path} \
	--use_QLora=True \
	--batch_size=4 \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=1 \
	--gradient_accumulation_steps=8 \
	--learning_rate=2e-5

echo '----------------------------- PRM model, memory efficient -------------------------------'
bash ./scripts/run_prm_examples.sh ${model_sft_step_full_path}  ${model_prm_lora_path} 2048
