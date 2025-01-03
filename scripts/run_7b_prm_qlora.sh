# 完整运行

# base_model_path='meta-llama/Meta-Llama-3-8B'
# base_model_path='xiaodongguaAIGC/xdg-llama-3-8B'
base_model_path='xiaodongguaAIGC/llama-3-debug'
deepspeed_config_name=./config/ds.json
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'
model_sft_step_lora_path=${output_path}'/sft_step_lora'
model_sft_step_full_path=${output_path}'/sft_step_full'


echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'


# # # stage: sft
sft_dataset_name='qgallouedec/prm800k'
model_pretrained_full_path=${base_model_path}
deepspeed ./ma-rlhf/sft_step.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=512 \
	--output_name=${model_sft_step_lora_path} \
	--use_QLora=False \
	--batch_size=8 \
	--use_flash_attention_2=False \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=2 \
	--gradient_accumulation_steps=4 \
	--learning_rate=1e-5

#  # merge SFT with lora
#  python ./ma-rlhf/merge_adapter.py \
#  	--base_model_name=${model_pretrained_full_path} \
#  	--model_name=${model_sft_step_lora_path} \
#  	--merged_model_name=${model_sft_step_full	_path}

# bash ./scripts/run_generation_examples.sh ${model_sft_full_path} 512
