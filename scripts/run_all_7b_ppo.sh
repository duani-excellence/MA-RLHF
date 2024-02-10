base_model_path='meta-llama/Llama-2-7b-hf'
deepspeed_config_name=./config/ds.json
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'
model_sft_lora_path=${output_path}'/sft_lora'
model_sft_full_path=${output_path}'/sft_full'
model_reward_model_lora_path=${output_path}'/reward_model_lora'
model_ppo_lora_path=${output_path}'/ppo_lora'
model_ppo_full_path=${output_path}'/ppo_full'


echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'

# # stage: sft
# sft_dataset_name='yahma/alpaca-cleaned'
# model_pretrained_full_path=${base_model_path}
# deepspeed ./ma-rlhf/sft.py \
# 	--dataset_name=${sft_dataset_name} \
# 	--model_name=${model_pretrained_full_path} \
# 	--seq_length=512 \
# 	--output_name=${model_sft_lora_path} \
# 	--use_QLora=True \
# 	--batch_size=8 \
# 	--use_flash_attention_2=True \
# 	--deepspeed_config_name=${deepspeed_config_name} \
# 	--num_train_epochs=1


# # merge SFT
# python ./ma-rlhf/merge_adapter.py \
# 	--base_model_name=${model_pretrained_full_path} \
# 	--model_name=${model_sft_lora_path} \
# 	--merged_model_name=${model_sft_full_path}


# echo '-------------------------------------------------------'
# date
# echo '-------------------------------------------------------'

# # stage reward model
# rm_dataset_name='Anthropic/hh-rlhf'
# deepspeed ./ma-rlhf/reward_model.py \
# 	--dataset_name=${rm_dataset_name} \
# 	--model_name=${model_sft_full_path} \
# 	--seq_length=256 \
# 	--batch_size=16 \
# 	--output_name=${model_reward_model_lora_path} \
# 	--use_QLora=True \
# 	--use_flash_attention_2=True \
# 	--deepspeed_config_name=${deepspeed_config_name} \
# 	--num_train_epochs=1


echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'
# stage ppo
rm_dataset_name='Anthropic/hh-rlhf'
deepspeed ./ma-rlhf/ppo.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--reward_model_name=${model_reward_model_lora_path} \
	--output_name=${model_ppo_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--batch_size=16 \
	--mini_batch_size=2 \
	--ppo_epochs=2 \
	--output_max_length=128 \
	--seq_length=256 \
	--gradient_accumulation_steps=4


# # merge PPO
# python ./ma-rlhf/merge_adapter.py \
# 	--base_model_name=${model_sft_full_path} \
# 	--model_name=${model_ppo_lora_path} \
# 	--merged_model_name=${model_ppo_full_path}


# echo '-------------------------------------------------------'
# date
# echo '-------------------------------------------------------'


# # generate result
# echo "------------------print sft result------------------"
# python ./ma-rlhf/generate.py \
# 	--model_name=${model_sft_full_path} \
# 	--prompt='give me a C++ code about quick sort?' \
# 	--max_new_tokens=128


# echo "------------------print ppo result------------------"
# python ./ma-rlhf/generate.py \
# 	--model_name=${model_ppo_full_path} \
# 	--prompt='how to make a bomb?' \
# 	--max_new_token=128


# echo '-------------------------------------------------------'
# date
# echo '-------------------------------------------------------'
