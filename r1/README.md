# New Training Project

Reproduce R1-like tarining pipeline, features

1. support Dense/MoE Base Model with `huggingface:Trasformer/PEFT/TRL/...`
2. support `DeepSpeed` Multi-GPU ZeRO-1/2/3 tranining
3. support DPO training with `TRL`
4. support GRPO RLVR training by `Verl`
5. support Agentic-RL Case


env: `ubuntu2204-pytorch2.8.0`

```bash
conda create -n llm python=3.11
conda activate llm
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

flash attention find in `https://github.com/Dao-AILab/flash-attention/releases`, and download `.whl`, and pip install:

```bash
pip nstall flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

```bash
pip install verl==0.7.0 torch==2.8.0 vllm==0.11.0
```
