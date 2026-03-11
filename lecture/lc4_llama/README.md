# Llama

本节内容学习 Decoder 优化组件，及长文本拓展策略

- ✅ ：必读
- 🌟 ：重点学习代码，最好能够独立手撕

## Notebook

带着以下问题学习 Llama

1. Llama 在 LLM 的历史地位
2. 模型如何算的更精确
3. 为何主流 LLM 模型要关注 long-context 能力

| 文件名 | 介绍 | 必读 |
| ------ | ---- | ---- |
| `RMSNorm.ipynb`       | 系统介绍 Normalization，深度分析RMSNorm的特点，并可视化对比 RMSNorm/LayerNorm归一化的几何特性，加分项推导RMSNorm的梯度。 | ✅ |
| `GroupedQueryAttention.ipynb`       | 理解KVCache后，设计一种低KVCache的注意力方案，从MQA到GQA, 做好精度-存储之间的 trade-off。从特征角度分析注意力头冗余吗？ | ✅🌟 |
| `SwiGLU.ipynb`       | 门控提供一种更精细化的特征选择 | ✅ |
| `RoPE.ipynb`       | 理想的位置编码在注意力计算的形式下，能够保证严格的相对距离表示。 | ✅🌟 |
| `Llama.ipynb`       | 聚合 RMSNorm、GQA、SwiGLU、RoPE组件，实现Llama模型前向计算 | ✅🌟 |
| `NTK-aware-RoPE.ipynb`       | 长文本能力体现模型在复杂语境中的信息处理能力。位置编码中的衰减性可以直观导致注意力加权系数。我们先推导 PI，分析PI缺陷，从而引出 NTK-RoPE实现动态插值，实现高频外推、低频内插 | ✅ |
| `Benchmark.ipynb`       | 测评预训练模型的基础case：选择题。能够测评出模型的知识 |      |

