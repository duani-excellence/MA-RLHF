# 流水线并行

## Part1：PP基础

辅助PP的组件。如重计算和用autograd 手动求导

| 代码                          | 功能描述                          | 必要 |
| ----------------------------- | --------------------------------- | ---- |
| `checkpoint_basic.py`         | torch api梯度检查点               | ✅    |
| `custom_gradient_backward.py` | 手动用 autograd 求导              | ✅    |
| `checkpoint_scratch.py`       | 手动实现检查点，实现重计算        | ✅    |
| `pipeline_parallel_basic.py`  | 多卡实现单数据 forward & backward | ✅    |

## Part2：PP优化

实现经典PP算法，gpipe必实现。逐步为dualpipe做准备。

| 代码                                | 功能描述                                                     | 必要 |
| ----------------------------------- | ------------------------------------------------------------ | ---- |
| `pipeline_parallel_gpipe.py`        | 实现micro-batch 批量F 批量B                                  | ✅    |
| `pipeline_parallel_pipe_dream.py`   | 实现 1F1B，降低中间变量存储，即时消化掉F的显存占用           | 🌟    |
| `pipeline_parallel_pipe_dream_2.py` | 较上增加更多的batch，循环队列管理 中间变量，限制F的数量。    | 🌟    |
| `zero_bubble.py`                    | 1F1B1W,  分离难点在于 要存储用于计算w的中间数据，增加显存。但实现了 计算-通信隐藏。（资源消耗不会消失，只会转移） | 🌟    |
| `zero_bubble_seperate_dx_dw.py`     | 以上实现不够优雅，可以存储求dw的函数，从而简化代码实现。     | 🌟    |

## Part3: DualPipe

前两个实现和理解都简单，后者官方实现更加标准和完整


| 代码                                | 功能描述                                                     | 必要                                                   |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `dualpipe_simplest.py`        | 手动实现 chimira schedule，不含网络 | 🌟 |
| `dualpipe_xdg.py`      | easy-dualpipe, 简化5个操作。亮点实现在于 对通信op进行管理，要计算时先wait()。dualpipe是为了：减少 “moe ep 通信重叠” 所导致的bubble | 🌟 |
| `dualpipe.py`          | （难，可选）官方实现标准dualpipe，代码更加完整和通用         | 🌟       |

dualpipe的变种 cut-in-half，可自行设计。不做补充了。



---

## Note

- [x] custom gradient to autograd.function().backward()
- [x] checkpoint & gradient checkpoint & scratch gradient checkpoint
- [x] pipeline parallel
- [x] gpipe: micro-batch pipeline parallel
- [x] pipe-dream: recompute trick to decrease memory: 1F1B
- [x] 1F1B, micro-batch = 8, ranks = 4
- [x] Zero-Bubble: compute-communication(dw,dx) overlap
- [x] dual-Pipe simplest
        - `dualpipe_simplest.py` 拆解自定义dualpipe为: F,FF,FB,BB,B 操作集合
        - `dualpipe_xdg.py` 依照自己设计的dualpipe, 来实现网络的前向和后向
- [x] dual-Pipe
- [ ] cut-in-half

1. 当前MoE类网络训练瓶颈是什么？ 
2. dualpipe解决了什么问题？讨论dualpipe的必要性
3. 如何通俗易懂的理解计算-通信重叠，实现对象是什么，输入输出是什么
4. dualpipe的schedule的设计思路
5. 通信重叠在原论文中称为1F1B, 0B1B, 0F1F 的重叠是否可以实现？