# xtrain

## Overview

minimal distribution training implementation from scratch

受 [open-infra-index](https://github.com/deepseek-ai/open-infra-index) 启发，为了掌握其所涉及的分布式训练技术栈，独立系统手撕关键的并行算法，简化实现以帮助更快理解 算法原理。

- 不依赖 `DeepSpeed` 和 `Megatron` 框架，从零实现 `5` 大类并行算法。
- 纯 `Pytorch` 手撕 `DP`、`TP`、`PP`、`CP`、`EP`分布式训练算法。
- 硬核手撕关键算法 `Backward` ，手撕分布式`gradient`和`adam`，不依赖`autograd`和现成优化器
- Step-by-step 手撕 `DP:ZeRO-3`、`TP:Llama`、`CP: RingAttention`、`PP: DualPipe`、`EP: Gshard`
- 硬核实现 MoE EP 1F1B 下的 **通信-计算重叠** 
- 不需要多卡环境，纯CPU GLOO backend可运行所有实例，无须 triton和cuda等基础

## Details

额外补充如数据、gradient、adam、io的相关组件的实现，以帮助并行算法的理解和实现。

在目录 `\lecture` 下各个章节有 `readme` 标注 `✅必要`, `🌟可选` 的代码。

| DP             | TP                | PP                 | CP                   | EP             |
| -------------- | ----------------- | ------------------ | -------------------- | -------------- |
| 分布式数据     | **col-parallel**  | **梯度检查点**     | Ring-AllReduce       | All2All        |
| **DP梯度**     | **row-parallel**  | **PP-basic**       | **Softmax**          | 异步All2All    |
| **分布式Adam** | **SwiGLU**        | **Gpipe**          | Online-Softmax       | **TopK 梯度**  |
| **ZeRO-1**     | **🔥GQA**          | **PipeDream**      | Ring-Softmax         | **SMoE**       |
| **ZeRO-2**     | **LMhead**        | **Zero-Bubble**    | **FlashAttention-2** | **🔥GShard**    |
| **🔥ZeRO-3**    | **Embedding**     | 🔥**Easy-Dualpipe** | **🔥Ring-Attention**  | **1F1B Basic** |
| 混合精度Adam   | **RMS-Norm**      | **DualPipe**       |                      | **🔥1F1B 重叠** |
| IO-load-save   | **🔥CrossEntropy** |                    |                      | V3-MoE结构     |

> [!NOTE]
>
> 1. 要求需要能熟练推导 Linear、Softmax、CrossEntropy 梯度
>
> 2. 代码简化实现，代码实例都可以运行，不保证效率和准确性。
>
> 3. Mac m系列芯片CPU可运行，禁用CUDA运行 `export CUDA_VISIBLE_DEVICES=-1 train.py`

## TODO

2期计划实现。不定期更新

- [ ] FP-8 training
- [ ] 训推分离
- [ ] Ray
- [ ] FlashMLA
- [ ] 推理 MoE 计算通信重叠
- [ ] ......

## Reference

[easy-dualpipe](https://github.com/dhcode-cpp/easy-dualpipe)

[dualpipe](https://github.com/deepseek-ai/DualPipe)

[cut-cross-entropy](https://github.com/dhcode-cpp/cut-cross-entropy-pytorch)

[cut-cross-entropy official]()

[online-softmax]()

[FlashAttention-v2](https://zhuanlan.zhihu.com/p/670085985)



