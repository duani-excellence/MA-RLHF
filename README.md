# MA-RLHF

MA-RLHF(Multiple Adapter-RLHF)  Project

- Deepspeed + TRL + QLoRA 
- LLaMA2-7B -> RLHF   In  Colab



## MA-RLHF Pipeline & Dataset

- `Pretrained`: IMDB
- `SFT`: Alpaca
- `Reward Model`: Anthropic-HH
- `RLHF `: Reward Model



## Installation

CUDA required

```
conda create -n llm python=3.10
conda activate llm
pip install -r requirements.txt
```

DeepSpeed Test

```
deepspeed ./test/test_QLoRA.py
```



## Quick Start

```
./scripts/run_all.sh
```



## Result

Environment 8x3090

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |


