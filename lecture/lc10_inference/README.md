# Inference

从推理服务角度，讲解 Inference 技巧。本 topic 实现 vLLM 里的关键技术。

本章代码仅个人根据 [vLLM技术报告](https://docs.vllm.ai/en/latest/community/meetups/#past-meetups) 复现各个功能点，与 vLLM 实现逻辑有一定出入。推理服务框架偏 system，会杂糅很多业务功能，在理解 vLLM 原理后，有两种不同代码学习模式：

1. 根据 vLLM 技术报告，凭借理解，逐步写出功能
2. 阅读 vLLM master 分支代码，并寻找 blog 阅读源码

- ✅ ：必读
- 🌟 ：重点学习代码，最好能够独立手撕

## Notebook

带着以下问题学习：

1. 分析推理、推理服务、集群推理服务差异？
2. vLLM 系统的关键组件有哪些？
3. 设计一个基于 Ray 的 P/D 分离系统，分析 PD 节点之间的 KV 传播，增加吞吐的 Latency 是必然的吗？
4. 分析 DeepSeek-V3 Inference 方案，计算推理成本（涉及硬件）

| 文件名                     | 介绍                                                                                                                                      | 必读 |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| `notebook/flashattention`  | 1. Online-softmax, 2. FlashAttention 3. FlashAttention-V2 4.FlashAttention-Backward                                                       | ✅🌟   |
| `Continue_Batching.ipynb`  | 实现最小的动态批服务，解决在线随机请求场景 batch-decoding 效率低的问题，从而提升吞吐率                                                    | ✅🌟   |
| `vLLM-PageKVCache.ipynb`   | 解决ContinueBatching中的 KVCache 管理问题，设计分页KVCache系统来提高 Cache 利用率。                                                       | ✅    |
| `vLLM-PageAttention.ipynb` | 设计 PageAttention Kernel(较难)                                                                                                           | ✅🌟   |
| `vLLM-V0.ipynb`            | 带 PageKVCache 与 vLLM-PageAttention 实现                                                                                                 |      |
| `Chunked-Prefill.ipynb`    | 实现 proj 搭便车和 PD batch 混合计算, 为 vLLM-V1  实现打下基础                                                                            | ✅    |
| `vLLM-V1.ipynb`            | 完成, 实现chunked prefill + PageAttention kernel。 运行命令：`python lecture/lc10_inference/vLLM-V1/server.py` 下可运行完整混合PD批次推理 | ✅    |
| `PD-Disaggreation.ipynb`   | 基于 Ray 实现 PD 节点分离, 前置要求学习 `ray`, TODO:分析 OpenRLHF/verl 单控制器                                                           |      |
|                            | TODO: Speculative Decoding                                                                                                                |      |

