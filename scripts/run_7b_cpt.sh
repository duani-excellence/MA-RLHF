base_model_path='meta-llama/Meta-Llama-3-8B'
# base_model_path='xiaodongguaAIGC/llama-3-debug'
deepspeed_config_name=./config/ds.json
output_path='./output'

model_pretrained_lora_path=${output_path}'/pretrained_lora'
model_pretrained_full_path=${output_path}'/pretrained_full'


echo '-------------------------------------------------------'
date
echo '-------------------------------------------------------'

# # # stage: second pretrained
pt_dataset_name='stanfordnlp/imdb'
deepspeed ./ma-rlhf/pretrained.py \
	--dataset_name=${pt_dataset_name} \
	--model_name=${base_model_path} \
	--seq_length=512 \
	--batch_size=4 \
	--output_name=${model_pretrained_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=1

# merge pretrained + LoRA = pretrained_lora
python ./ma-rlhf/merge_adapter.py \
	--base_model_name=${base_model_path} \
	--model_name=${model_pretrained_lora_path} \
	--merged_model_name=${model_pretrained_full_path}
