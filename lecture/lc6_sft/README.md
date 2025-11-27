# SFT/Application

学习 SFT 训练，本课件将是实操项目的一个重要的前置项目

- ✅ ：必读
- 🌟 ：重点学习代码，最好能够独立手撕

## Notebook

带着以下问题学习：

1. 描述指令、指令微调、指令跟随三者定义和联系
2. 全参微调和LoRA微调之间的差异
3. 根据你的业务收集种子数据（如200条），根据 AI 合成产生 1W+ 以上数据，并对模型实施微调
4. CoT、微调、RAG、Agent、AgenticRL 哪些是借用了外部能力、哪些是依靠内部能力

| 文件名 | 介绍 | 必读 |
| ------ | ---- | ---- |
| `Supervised_Finetuning_Dataset.ipynb`       | 基于Pytorch，手撕SFT Dataset, 实现 Messages 格式、对话模版、Collate、dataloader。并且可以取batch | ✅ |
| `Supervised_Finetuning_PyTorch.ipynb`       | 基于Pytroch，调用现成模型，手动写训练函数，微调后模型成功。遵循对话模版，能够理解提示词并生成出合理的结果 | ✅ |
| `Supervised_FineTuning_transformers_Qwen3.ipynb`       | 基于Transformers库，调用dataset库预处理数据；实现SFT手动版本、Trainer版本、以及trl::SFTTrainer 版本 | ✅🌟 |
| `LoRA.ipynb`       | LoRA 原理推导，低秩矩阵究竟在 fitting 什么目标，LoRA 值的深究的点在于为什么rank能做到那么低？初始化策略是什么？LoRA如何推导梯度？在实操过程中需要对比LoRA微调和全参微调之间的差别。 | ✅ |
| `Prompt_Enginerring.ipynb`       | TODO |      |
| `RAG.ipynb`       | TODO |      |
| `Embedding.ipynb`       | TODO |      |
| `ReAct.ipynb`       | TODO |      |
| `LLM_as_a_Judge.ipynb`       | TODO |      |
| `SimpleEval.ipynb`       | TODO |      |





## Note

微调基于 Qwen3 预训练模型, Alpaca 数据集

2. SFT Dataset: `Supervised_Finetuning_Dataset.ipynb`
3. full finetuning PyTorch: `Supervised_Finetuning_PyTorch.ipynb`
4. full finetuning Huggingface: `Supervised_FineTuning_transformers_Qwen3.ipynb`
5. LoRA finetune
6. QLoRA finetune

## 提示词工程

6. prompt engineering
7. CoT
8. alpaca

## 外部能力

9. embedding
10. RAG
11. agent ReAct

## 测评

12. benchmark evaluation
13. llm as a judge
