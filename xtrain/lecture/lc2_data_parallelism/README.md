# 数据并行

| 代码                     | 功能描述                                                     | 必要 |
| ------------------------ | ------------------------------------------------------------ | ---- |
| `torch_ddp_train.py`     | 调用 torch 现成 API 实现自定义模型和数据的训练，和梯度参数更新 | ✅    |
| `distributed_dataset.py` | 手撕并行数据类，用于在多卡分发数据                           | ✅    |
| `torch_ddp.py`           | 手撕 DP 下的梯度更新与 torch实现一致。各rank独自backward，再reduce梯度。⚠️非loss reduce后再backward。 | ✅    |

