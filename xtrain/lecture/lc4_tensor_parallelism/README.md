# 张量并行

## Part1：行列并行

以行列并行实现线性层forward & backward

| 代码                             | 功能描述                               | 必要 |
| -------------------------------- | -------------------------------------- | ---- |
| `tensor_parallel_easy.py`        | 单卡版本                               | ✅    |
| `tensor_parallel.py`             | 分布式行列并行，前向                   | ✅    |
| `custom_gradient.py`             | 手撕定义 autograd 函数                 | ✅    |
| `distributed_custom_gradient.py` | 在反向函数里增加 分布式通信操作        | ✅    |
| `col_parallel_linear.py`         | 列并行前后向实现，backward时reduce操作 | ✅    |
| `row_parallel_linear.py`         | 行并行前后向操作，foward时reduce       | ✅    |

⚠️切分配套实现：合并参数，或者 shard io。 其切分较 zero 更合理（但是需要针对层设计 切分策略，zero则对任意尺寸/功能参数都可以flatten切分）

## Part2：模块并行

实现难度较高，能够实现前两个就差不多了。

| 代码            | 功能描述                                                     | 必要 |
| --------------- | ------------------------------------------------------------ | ---- |
| `mlp.py`        | 组合行列并行类实现MLP/SwiGLU，受激活函数影响，其切分策略固定。 | ✅    |
| `attention.py`  | GQA 较为复杂（难）。1.分组组网。2.实现部分Wq，Wk，Wv散步在各组内，组内Wk，Wv参数一致。3. head-parallel 4.dk,dv, 梯度reduce计算 | 🌟    |
| `embedding.py`  | 分为词表并行（难）和维度并行（简单）。backward 注意梯度需要 combine。词表并行需要注意 idx-offset | 🌟    |
| `lm_head.py.py` | 词表并行，融合cross entropy优化反向函数（难）。词表并行需要注意 idx-offset。 参考：[cut-cross-entropy](https://github.com/dhcode-cpp/cut-cross-entropy-pytorch) | 🌟    |
| `rms_norm.py`   | 各GPU参数一样，backward 梯度（难），建议用deepseek等辅助推导，梯度的reduce 在backward里实现。 | 🌟    |
| `decoder.py`    | 定义decodeblock，组合上述mlp、attention、rms_norm            | 🌟    |
| `model.py`      | 定义 TP model类 `XtrainModel` 前后向均可以运行               | 🌟    |

🌟 表示加分项
