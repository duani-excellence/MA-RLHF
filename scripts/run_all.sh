

# # wandb
# #

# base_model_path='meta-llama/Llama-2-7b-hf'


# deepspeed_config_name=./../config/ds.json
# text_dataset_name='./../dataset/med_qa_textbook'
# pt_dataset_name='./../dataset/second_pretrained_datasets'
# rm_dataset_name='./../datasets/medical_reward'

# output_path='/data/'

# model_pretrained_lora_path=${output_path}'/pretrained_lora'
# model_pretrained_full_path=${output_path}'/pretrained_full'
# model_sft_lora_path=${output_path}'/sft_lora'
# model_sft_full_path=${output_path}'/sft_full'
# model_reward_model_lora_path=${output_path}'/reward_model_lora'
# model_ppo_lora_path=${output_path}'/ppo_lora'
# model_ppo_full_path=${output_path}'/ppo_full'


# # stage: second pretrained
# # pt_dataset_name='./../second_pretrained_datasets'
# deepspeed ./../rlhf/pretrained.py \
# 	--dataset_name=${pt_dataset_name} \
# 	--model_name=${base_model_path} \
# 	--seq_length=16 \
# 	--batch_size=8 \
# 	--output_name=${model_pretrained_lora_path}\
# 	--use_QLora=True \
# 	--use_flash_attention_2=True \
# 	--deepspeed_config_name=${deepspeed_config_name} \


# # merge pretrained + LoRA = pretrained_lora
# python ./../rlhf/merge_adapter.py \
# 	--base_model_name=${base_model_path} \
# 	--model_name=${model_pretrained_lora_path} \
# 	--merged_model_name=${model_pretrained_full_path}


# # stage: sft
# deepspeed ./../rlhf/sft.py \
# 	--dataset_name=${rm_dataset_name} \
# 	--model_name=${model_pretrained_full_path} \
# 	--seq_length=16 \
# 	--output_name=${model_sft_lora_path} \
# 	--use_QLora=True \
# 	--batch_size=8 \
# 	--use_flash_attention_2=True \
# 	--deepspeed_config_name=${deepspeed_config_name} \


# # merge SFT
# python ./../rlhf/merge_adapter.py \
# 	--base_model_name=${model_pretrained_full_path} \
# 	--model_name=${model_sft_lora_path} \
# 	--merged_model_name=${model_sft_full_path}


# # stage reward model
# deepspeed ./../rlhf/reward_model.py \
# 	--dataset_name=${rm_dataset_name} \
# 	--model_name=${model_sft_full_path} \
# 	--seq_length=8 \
# 	--batch_size=8 \
# 	--output_name=${model_reward_model_lora_path} \
# 	--use_QLora=True \
# 	--use_flash_attention_2=True \
# 	--deepspeed_config_name=${deepspeed_config_name} \


# # stage ppo
# deepspeed ./../rlhf/ppo.py \
# 	--dataset_name=${rm_dataset_name} \
# 	--model_name=${model_sft_full_path} \
# 	--reward_model_name=${model_reward_model_lora_path} \
# 	--seq_length=8 \
# 	--output_name=${model_ppo_lora_path} \
# 	--use_QLora=True \
# 	--use_flash_attention_2=True \
# 	--deepspeed_config_name=${deepspeed_config_name} \
# 	--batch_size=16 \
# 	--mini_batch_size=2 \
# 	--ppo_epochs=1 \
# 	--output_max_length=128 \


# # merge PPO
# python ./../rlhf/merge_adapter.py \
# 	--base_model_name=${model_sft_full_path} \
# 	--model_name=${model_ppo_lora_path} \
# 	--merged_model_name=${model_ppo_full_path}


# # generate result
# echo "------------------print ppo result------------------"
# python ./../rlhf/generate.py \
# 	--model_name=${model_ppo_full_path} \
# 	--prompt='hello world?' \
# 	--max_new_token=128

# python ./../rlhf/generate.py \
# 	--model_name=${model_ppo_full_path} \
# 	--prompt='hello world?' \
# 	--max_new_token=128
