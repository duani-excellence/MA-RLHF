# GPT

本章节学习 Decoder Transfomer 模型，全套代码要求手撕

- ✅ ：必读
- 🌟 ：重点学习代码，最好能够独立手撕

## Notebook

带着以下问题学习 GPT 

1. 为什么主流的学习方式由监督学习转变为预训练？
2. GPT 预训练学习到什么？
3. 预训练能够进行 ScaleUp 的前提是什么？

| 文件名                        | 介绍                                                         | 必读 |
| ----------------------------- | ------------------------------------------------------------ | ---- |
| `GPT-2.ipynb` | 实现Model、数据封装、训练、推理功能 | ✅🌟   |
| `GELU.ipynb` | 实现 GELU 推导 |      |
| `Pre-Normalization.ipynb`     | 分析 Pre-Normalization 为什么有效               | ✅    |
| `Perplexity.ipynb`             | Next-Token-Prediction任务的性能指标                          | ✅   |
| `BPE-Tokenizer.ipynb`          | 一步步实现通用分词器                                         | ✅ |
| `LM_dataset.ipynb`               | 根据GPT-2 WebText 封装 dataloader，并可以加载batch数据：input_ids, labels, attention_mask | ✅   |
| `KVCache.ipynb`              | 缓存历史 KV，减少Proj和 Attention 的重复计算，从而提高 Inference 推断效率 | ✅🌟   |
| `in_context_learning_inference.ipynb`              | 引入模版提示词技巧，改变输入从而改变输出。                   | ✅    |



