# 基础通信

## Part1：基础环境搭建

基于`torch.distributed` 的 `GLOO` 来实现分布式环境启动。

| 代码                  | 功能描述       | 必要 |
| --------------------- | -------------- | ---- |
| `multi_processing.py` | 多线程启动代码 | ✅    |
| `initial.py`          | 初始化环境GLOO | ✅    |
| `group.py`            | 手动组群       | ✅    |
| `device_mesh.py`      | 2D自动组群     | ✅    |

## Part2：P2P 通信

重要，必会手动写 P2P。

| 代码                 | 功能描述                 | 必要 |
| -------------------- | ------------------------ | ---- |
| `p2p.py`             | (阻塞式)点到对通信Tensor | ✅    |
| `p2p_async.py`       | 异步点到对通信Tensor     | ✅    |
| `p2p_object.py`      | 传递对象，非Tensor       |      |
| `p2p_op.py`          | 环形通信解死锁           | ✅    |
| `p2p_cuda_stream.py` | 未实现                   |      |

## Part3：通信操作

重要，必会手动基于 P2P 写各种操作，会根据情况设计 同步/异步 通信解死锁。

| 代码                     | 功能描述                           | 必要 |
| ------------------------ | ---------------------------------- | ---- |
| `fun_gather.py`          | 收集，调用API和手撕P2P实现         | ✅    |
| `fun_scatter.py`         | 分散，调用API和手撕P2P实现         | ✅    |
| `fun_broadcast.py`       | 广播，调用API和手撕P2P实现         | ✅    |
| `fun_reduce.py`          | 规约，调用API和手撕P2P实现         | ✅    |
| `reduce_op.py`           | 规约操作实例                       | ✅    |
| `func_reduce_scatter.py` | 复合操作，调用API和手撕P2P实现     | ✅    |
| `fun_all2all.py`         | all2all，调用API                   | ✅    |
| `fun_ring_allreduce.py`  | 手撕P2P实现环形通信，解死锁        | 🌟    |
| `fun_all2all_scratch.py` | 手撕P2P实现All2All同步通信，解死锁 | 🌟    |

🌟表示加分项。

另外 `lecture/lc7_MoE/fun_all2all_ascyn.py` 实现了异步 All-to-All，用于通信-计算重叠。

## Part4. 其他

| 代码          | 功能描述       | 必要 |
| ------------- | -------------- | ---- |
| `torchrun.py` | torchrun启动   |      |
| `store.py`    | 共享变量       |      |
| `profile.py`  | torch 性能分析 | ✅    |

