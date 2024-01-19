# MA-RLHF

MA-RLHF(Multiple Adapter-RLHF)  Project

- Deepspeed + TRL + QLoRA
- LLaMA2-7B -> RLHF   In  Colab



## MA-RLHF Pipeline & Dataset

- `Pretrained`: IMDB
- `SFT`: Alpaca
- `Reward Model`: Anthropic-HHRLHF
- `PPO`: Anthropic-HHRLHF



## Installation

CUDA required

```bash
conda create -n llm python=3.9
conda activate llm
pip install -r requirements.txt
```

DeepSpeed Test

```bash
deepspeed ./test/test_QLoRA.py
```

Setting Environment

```bash

export WANDB_API_KEY={YOU_WANDB_TOKEN} # from https://wandb.ai/authorize
# export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_P2P_DISABLE="1" # for 3090/4090
# export NCCL_IB_DISABLE="1"  # for 3090/4090

```

## Quick Start

```bash
./scripts/run_step.sh
```

### 1. Pretrained(Optional)

### 2. SFT

### 3. Reward Model

### 4. PPO

### 5. Merge Adapter

### 6. Generate Test

### 7. Custom Data


## Result

Environment 8x3090

|     |     |     |
| --- | --- | --- |
|     |     |     |
|     |     |     |
|     |     |     |

- 8xA800
- Basic Model LLaMA2-7B 1 epochs
- Pretrained : IMDB 22k Time: 20min
- SFT : Alpaca 52k Time: 20min
- Reward Model: HHRLHF 160k Time 1h20min
- PPO :         HHRLHF 160k Time 1h
- total  : 3h


- Basic Model LLaMA2-7B 1 epochs
- 
