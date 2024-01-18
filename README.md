# MA-RLHF

MA-RLHF(Multiple Adapter-RLHF)  is a low-cost and efficient large language model training system

Feature：

- Deepspeed + TRL + QLoRA + Flash-Attntion 2
- LLaMA2-13B



## MA-RLHF Pipeline & Dataset

- `Pretrained`: imdb
- `SFT`: yahma/alpaca-cleaned
- `Reward Model`: Anthropic/hh-rlhf
- `PPO`: Anthropic/hh-rlhf



## Installation

Git Clone MA-RLHF

```
git clone git@github.com:dhcode-cpp/MA-RLHF.git
cd MA-RLHF
```

Create Dev Environment

```bash
conda create -n llm python=3.11
conda activate llm
pip install -r requirements.txt
pip install flash-attn
```

Setting Environment

```bash
export WANDB_API_KEY={YOU_WANDB_TOKEN} # from https://wandb.ai/authorize
# export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_P2P_DISABLE="1" # for GPU 3090/4090
# export NCCL_IB_DISABLE="1"  # for GPU 3090/4090
```

DeepSpeed Test

```bash
deepspeed ./test/test_QLoRA.py
```

- Deepspeed config json is `./config/ds.json`



## Result

Environment 8x3090

|            | 8xA100    | 8xA100      | 8xA100     | 8xA100      | 8x3090     | 8x3090     |
| ---------- | --------- | ----------- | ---------- | ----------- | ---------- | ---------- |
| VRAM       | 40GB      | 40GB        | 80GB       | 80GB        | 24GB       | 24GB       |
| Model      | LLaMA2-7B | LLaMA-2-13B | LLaMA-2-7B | LLaMA-2-13B | LLaMA-2-7B | LLaMA-2-7B |
| ZeRO       | 1         | 1           |            |             |            |            |
| Epochs     | 1         | 1           |            |             |            |            |
| Pretrained | 20min     | x           |            |             |            |            |
| SFT        | 20min     | x           |            |             |            |            |
| Reward     | 1h20min   | x           |            |             |            |            |
| PPO        | 1h30min   | x           |            |             |            |            |
| Total      | 3h30min   | x           |            |             |            |            |



## Quick Start

运行全流程

```bash
./scripts/run_all.sh
```

### 0. run path

```bash
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
```

### 1. Pretrained(Optional)

```bash
# stage: second pretrained
pt_dataset_name='imdb'
deepspeed ./ma-rlhf/pretrained.py \
	--dataset_name=${pt_dataset_name} \
	--model_name=${base_model_path} \
	--seq_length=512 \
	--batch_size=16 \
	--output_name=${model_pretrained_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--deepspeed=${deepspeed_config_name} \
	--num_train_epochs=1
```

### 2. SFT

```bash
sft_dataset_name='yahma/alpaca-cleaned'
model_pretrained_full_path=${base_model_path}
deepspeed ./ma-rlhf/sft.py \
	--dataset_name=${sft_dataset_name} \
	--model_name=${model_pretrained_full_path} \
	--seq_length=512 \
	--output_name=${model_sft_lora_path} \
	--use_QLora=True \
	--batch_size=16 \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=1
```

### 3. Reward Model

```bash
# stage reward model
rm_dataset_name='Anthropic/hh-rlhf'
deepspeed ./ma-rlhf/reward_model.py \
	--dataset_name=${rm_dataset_name} \
	--model_name=${model_sft_full_path} \
	--seq_length=512 \
	--batch_size=8 \
	--output_name=${model_reward_model_lora_path} \
	--use_QLora=True \
	--use_flash_attention_2=True \
	--deepspeed_config_name=${deepspeed_config_name} \
	--num_train_epochs=1
```

### 4. PPO

```bash
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
	--batch_size=4 \
	--mini_batch_size=2 \
	--ppo_epochs=1 \
	--output_max_length=256 \
	--seq_length=512
```

### 5. Merge Adapter

```bash
# merge PPO
python ./ma-rlhf/merge_adapter.py \
	--base_model_name=${model_sft_full_path} \
	--model_name=${model_ppo_lora_path} \
	--merged_model_name=${model_ppo_full_path}
```

### 6. Generate Test

```bash
echo "------------------print ppo result------------------"
python ./ma-rlhf/generate.py \
	--model_name=${model_ppo_full_path} \
	--prompt='how to make a bomb?' \
	--max_new_token=128
```



## TODO

- [ ] Add LLM Performance + scaling
- [ ] evaluation module
- [ ] DPO
- [ ] Rejection sampling
- [ ] Custom Data Tutorial
- [ ] Video For MA-RLHF 



## About Me

微信 ：xiaodongguaAIGC

微信公众号：手撕LLM

知乎/小红书：小冬瓜AIGC

git: decode-cpp

