# 专家并行

## Part1：SMoE基础

| 代码                | 功能描述                                                     | 必要 |
| ------------------- | ------------------------------------------------------------ | ---- |
| `some_forward.py`   | 单机前向MoE计算，单卡dispatch-combine，增加负载均衡损失      | ✅    |
| `smoe_backward.py`  | 较上增加backward，专家梯度，gate梯度，负载均衡梯度 实现。实现backward的目的是为了感受 反向也做 dispatch-combine | ✅    |
| `top_k_gradient.py` | 不可导top-k在torch的实现，用于求gate的梯度                   | ✅    |

## Part2：EP

| 代码        | 功能描述                                                     | 必要 |
| ----------- | ------------------------------------------------------------ | ---- |
| `gshard.py` | 专家并行最经典算法，各rank除了 expert不一样，其他参数都一样。将smoe进行多卡实现，特别实现all-to-all时需要较多中间值辅助数据传输。1. 手动实现了同步/异步All-to-All交互`list[tensor]` 2. forward保留 all-to-all map和中间值，反向遵循 all-to-all map。 3. forward/backward，其EP均是 dispatch->combine顺序。 | ✅    |

## Part3：通信-计算重叠

| 代码                   | 功能描述                                                     | 必要 |
| ---------------------- | ------------------------------------------------------------ | ---- |
| `fun_all2all_ascyn.py` | All-to-All 异步实现（必须）。case如 先在后台执行发送，无须等待传输完成，同时可以安排做计算。 | ✅    |
| `overlapped_1F1B.py`   | 基于gshard。一个block要处理两份数据，一个用于Forward，一个求Backward。1. 这个Block的step实现非常复杂，其建立在能够异步做All-to-All的基础上。2. 通信计算重叠的代价就是要增加一些中间值保存，我们定义`OverlappedOp` 辅助 all-to-all 通信，分离 expert的计算。 3.修改 F(mlp,attn)B(attn,mlp) 操作为 F(mlp)B(attn)F(attn)B(mlp)， 此时可以将all-2-all的dispatch/combine独立通信，不占用计算时间。 | 🌟    |

通信计算重叠不限定1F1B改造。前向时也可以实现，前提是2-micro-batch输入。事实上comm-comp overlap就是要求有2-micro-batch输入。

更复杂的实现 [deepep](https://github.com/deepseek-ai/DeepEP)



## Note

- [x] sMoE forward
- [x] top-k-gradient
- [x] sMoE backward
- [x] All-to-All
- [x] GShard forward/backward
- [x] DeepSeek-V3 MoE: 实现网络结构 + Loss (非 EP)
- [x] All-to-All-asycn
- [x] Computation-Communication Overlapped.
        - 双流流向实现通信掩盖
        - 分析 DualPipe 各操作的实现
        - 分析 Training, Prefilling, Decoding 阶段的重叠方案
        - 分析 DeepEP 库实现
- [] F0F1 Comp-Comm Overlapped