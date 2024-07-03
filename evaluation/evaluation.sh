MODEL_PT='./output/pt_full'
MODEL_SFT='./output/sft_full'
MODEL_RLHF='./output/rlhf_full'

# # 第一次运行生成
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./evaluation/vllm_generation.py \
# 	--model_name=${MODEL_SFT} \
# 	--output_name=./output/result_vllm_SFT  \
# 	--max_output_tokens=512 \
# 	--batch_size=2 \
# 	--num_gpus=4 &


# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluation/vllm_generation.py \
# 	--model_name=${MODEL_RLHF} \
# 	--output_name=./output/result_vllm_RLHF  \
# 	--max_output_tokens=512 \
# 	--batch_size=2 \
# 	--num_gpus=4 &

# # 第二次运行评估
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./evaluation/vllm_llama_guard.py \
# 	--model_name=meta-llama/Meta-Llama-Guard-2-8B \
# 	--dataset_name=./output/result_vllm_SFT \
# 	--output_name=./output/result_vllm_SFT_llamaguard  \
# 	--max_output_tokens=1 \
# 	--batch_size=256 \
# 	--num_gpus=4 &


# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluation/vllm_llama_guard.py \
# 	--model_name=meta-llama/Meta-Llama-Guard-2-8B \
# 	--dataset_name=./output/result_vllm_RLHF \
# 	--output_name=./output/result_vllm_RLHF_llamaguard  \
# 	--max_output_tokens=1 \
# 	--batch_size=256 \
# 	--num_gpus=4


# # 第三次运行pretrained, 不过没什么意义
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./evaluation/vllm_generation.py \
# 	--model_name=${MODEL_PT}  \
# 	--output_name=./output/result_vllm_pt  \
# 	--max_output_tokens=512 \
# 	--batch_size=2 \
# 	--num_gpus=4

# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./evaluation/vllm_llama_guard.py \
# 	--model_name=meta-llama/Meta-Llama-Guard-2-8B \
# 	--dataset_name=./output/result_vllm_pt \
# 	--output_name=./output/result_vllm_pt_llamaguard  \
# 	--max_output_tokens=1 \
# 	--batch_size=256 \
# 	--num_gpus=4
