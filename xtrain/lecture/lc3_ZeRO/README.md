# ZeRO

ZeRO本质仍然是DP

## Adam 基础

| 代码                  | 功能描述                                   | 必要 |
| --------------------- | ------------------------------------------ | ---- |
| `adam.py`             | 手撕 adam optimizer                        | ✅    |
| `distributed_adam.py` | 分布式Adam。保证梯度/参数/优化器参数一致。 | ✅    |

## ZeRO

本质在于 Shard 数据，而在计算层面，即算即取。

切分，在于将数据 flatten 化，而非矩阵层面上切分。

| 代码            | 功能描述                                                     | 必要 |
| --------------- | ------------------------------------------------------------ | ---- |
| `adam_zero1.py` | 优化器参数切分，各自更新部分权重后，进行 all-gather 同步一致性参数 | ✅    |
| `adam_zero2.py` | 手撕`loss.backward`, 手撕Backward Reduce-Scatter到各个GPU上  | ✅    |
| `adam-zero3.py` | 参数切分，自定义zero3前向，反向时不需要all-gather参数        | ✅    |

## 其他

配套zero-3关联实现

| 代码                     | 功能描述                                                     | 必要 |
| ------------------------ | ------------------------------------------------------------ | ---- |
| `adam_mix_precision.py`  | 混合精度训练                                                 | 🌟    |
| `zero_io.py`             | 也可以将完整的模型进行切分`model.shared_model()`。 类似保存zero-3的checkpoint。 |      |
| `zero_io_shared_load.py` | 模型加载一个超大的模型case。云端大模型分块存储。根据切分好的模型，定义shard 模型`SharedToyModel`来加载（减少显存），逐层从 cpu load数据，再传输到 GPU。 |      |