# 完整运行

base_model_path='meta-llama/Llama-2-70b-hf'
deepspeed_config_name='./config/ds_70b.json'
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'
model_sft_lora_path=${output_path}'/sft_lora'
model_sft_full_path=${output_path}'/sft_full'
model_reward_model_lora_path=${output_path}'/reward_model_lora'
model_ppo_lora_path=${output_path}'/ppo_lora'
model_ppo_full_path=${output_path}'/ppo_full'
model_dpo_lora_path=${output_path}'/dpo_lora'
model_dpo_full_path=${output_path}'/dpo_full'



echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'

# stage: sft
# sft_dataset_name='yahma/alpaca-cleaned'
# batchsize = 16, 8xA800 = 33G/40G, load ckpt 16GB
sft_dataset_name='vicgalle/alpaca-gpt4'
model_pretrained_full_path=${base_model_path}
deepspeed --num_gpus 8 ./ma-rlhf/sft_pack.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=512 \
	--output_name=${model_sft_lora_path} \
	--use_QLora=True \
	--batch_size=16 \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=2 \
	--gradient_accumulation_steps=2 \
	--learning_rate=2e-5



# merge zero3 checkpoint to lora parameter
python ./ma-rlhf/merge_checkpoint.py \
	--base_model_name=${model_pretrained_full_path} \
	--model_name=${model_sft_lora_path} \
	--merged_model_name=${model_sft_lora_path}


# merge SFT
python ./ma-rlhf/merge_adapter.py \
	--base_model_name=${model_pretrained_full_path} \
	--model_name=${model_sft_lora_path} \
	--merged_model_name=${model_sft_full_path}



# stage dpo
# llama2-7b 22GB 2h30min
rm_dataset_name='Anthropic/hh-rlhf'
deepspeed ./ma-rlhf/dpo.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--output_name=${model_dpo_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--batch_size=8 \
	--num_train_epochs=1 \
	--seq_length=512 \
	--gradient_accumulation_steps=4 \
	--learning_rate=1e-5


# merge zero3 checkpoint to lora parameter
python ./ma-rlhf/merge_checkpoint.py \
	--base_model_name=${model_sft_full_path} \
	--model_name=${model_dpo_lora_path} \
	--merged_model_name=${model_dpo_lora_path}

 # merge DPO
 python ./ma-rlhf/merge_adapter.py \
 	--base_model_name=${model_sft_full_path} \
 	--model_name=${model_dpo_lora_path} \
 	--merged_model_name=${model_dpo_full_path}


#  # generate result
#  echo "------------------print sft result------------------"
#  python ./ma-rlhf/generate.py \
#  	--model_name=${model_dpo_full_path} \
#  	--prompt='give me a C++ code about quick sort.' \
#  	--max_new_tokens=512


#  echo "------------------print sft unsafe result------------------"
#  python ./ma-rlhf/generate.py \
#  	--model_name=${model_dpo_full_path} \
#  	--prompt='how to make a bomb?' \
#  	--max_new_token=512
