# DeepSeek-V3

学习 MoE 历史关键模型，再从 Inference 导向角度理解各个模块的优化动机

- ✅ ：必读
- 🌟 ：重点学习代码，最好能够独立手撕

## Notebook

带着以下问题学习 DeepSeek-V3

1. 低成本训练和低成本推理哪个更加重要
2. 从硬件角度思考各个组件的优化成效（存储、计算、通信）

| 文件名 | 介绍 | 必读 |
| ------ | ---- | ---- |
| `Mixture-of-Experts.ipynb`       | 引入集成学习思想，丰富特征表达模式。MoE、SMoE和Dispatch-Combine计算模式 | ✅ 🌟 |
| `Load_Balance.ipynb`       | 手动设计一个极简的负载均衡策略，根据sMoE引入衡量负载均衡的指标。最后实现一个应用最广的 SwitchTransformer负载均衡 | ✅ |
| `DeepSeek-MoE.ipynb`       | 实现 shared-expert、router-expert计算、序列负载均衡、修正门控权重 | ✅🌟 |
| `Multi_Latent_Attention.ipynb`       | MLA为什么能将特征维度压的非常低，Cache量压缩到原有的约10%以内，值得思考的是MLA组件学习了 low-rank 的特征表达，这一特征是在预训练过程就能学会的，而降维视角MLA与MQA又有显著差异。另外MLA有特性：位置编码分离、KV位置编码单头、c-cache、权重矩阵吸收等 | ✅🌟 |
| `YaRN.ipynb`       | NTK-RoPE效果不好的原因是，保高频并不严格，YaRN分段严格保高频外推。 | ✅ |
| `Multi_Token_Prediction.ipynb`       | 引入 RNN 头做递归式next-token-prediction。在完整模型视角实现了 next-N-token-prediction。NTP的训练和推理内核并没有变化。能够手写出NNTP推理模型，也就意味着实现了“并行解码”。在MTP学习任务下，学习到一种新的特征模式：时序特征。 | ✅ 🌟 |
| `DeepSeek-V3.ipynb`       | 集成DeepSeek-MoE、MLA、YaRN、序列负载均衡 组件实现 V3 的前向计算 | ✅ |
| `top-k_backward.ipynb`       | 手撕 Torch.Topk backward | ✅ |

