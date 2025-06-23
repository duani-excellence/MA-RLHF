# 上下文并行

## Part1：softmax基础

| 代码                     | 功能描述                                      | 必要 |
| ------------------------ | --------------------------------------------- | ---- |
| `online_softmax.py`      | 批量数据块 online-softmax实现                 | ✅    |
| `ring_online_softmax.py` | 分布式版本 online-softmax，环形通信需解死锁。 | ✅    |
| `softmax_gradient.py`    | 用于推导 ring-attention backward              |      |

## Part2：上下文并行

| 代码                 | 功能描述                                                     | 必要               |
| -------------------- | ------------------------------------------------------------ | ------------------ |
| `ring_attention.py`  | 1. 看成分布式版本的 FlashAttention-V2，有V2基础再推ring-attention；2.各gpu参数一致，数据按seq维度切分后scatter到各GPU上； 3.定义环形通性组件。 4.实现前向计算，注意中间变量的保存； 5.backward技巧较多，注意dq，dk，dv块累加计算；6. dX的计算需要从dQ，dK，dV分支传递回来；7.KV通信均衡，块计算不均衡 | （✅前向）（🌟反向） |
| `compute_balance.py` | Stripe Attention，调整 KV 顺序，保证块计算均衡；由于仅调整 KV 序列，即在ring-attention 外围增加 mappping 操作。由于不涉及到新的算法，不实现。 | 🌟                  |



----

## Note

- [x] online-softmax
- [x] Ring online-Softmax
- [x] Ring Flash Attention V2 Forward
- [x] Ring Flash Attention V2 Backward
- [x] Striped Attention: compute balance

在上下文并行方法中, 最关键和最复杂的算法在于 Attention 的并行化, 如果能推导 Flash Attention-V2 就能实现 Ring Attention

而在其他的模块中, 并没有“序列建模”，即token之间是独立的 , 则便于“序列并行”, 分析关键层的序列并行实现, 均可以看成是"数据并行"模式

1. MLP: 直接前向计算, 看作是“数据并行“, 仅需要对梯度做All-Reduce即可完成一致性更新
2. RMS_Norm: 反向要all-reduce
3. Embedding: 需要注意 embedding 梯度的聚合

对于 Decoder-Only 的注意力, 仅“下三角”的块注意力需要计算, 所以会出现 计算 不均衡情况, 即rank 0 的计算量 远低于最后一个rank, 通信仍然是正常的。那么可以参考 `DistFlashAttn`/`StripedAttention`, 改变序列顺序, 从而均衡计算量, 该算法利用 “块注意力计算的无序性”, 即改变序列顺序, 不影响 块注意力计算, 比如"Q4 [K1, K3], [K2, K4]" 计算的注意力和"Q4 [K1, K2], [K3, K4]" 最终注意力输出是相同的。

示例1, 标准 block-wise attention, X12 表示 第1块Q和第2块K 所计算的注意力分数块
```
rank 0: X00, X01, X02, X03
rank 1: X10, X11, X12, X13
rank 2: X20, X21, X22, X23
rank 3: X30, X31, X32, X33
```

示例2, Decode-only block-wise attention, "-" 表示可以忽略的块注意力计算
```
rank 0: X00,  -,   -,   -
rank 1: X10, X11,  -,   -
rank 2: X20, X21, X22,  -
rank 3: X30, X31, X32, X33
```


示例3, 调整序列顺序 block-wise attention, 先调换 输入序列块 1 <--> 3, 我们需要计算的注意力块如下
```
rank 0: X00,  -,   -,   -
rank 1: X30, X31, X32, X33  
rank 2: X20, X21, X22,  -
rank 3: X10, X11,  -,   -
```

示例4, 我们仍观察到每个rank计算量仍不均衡, 我们假设有两个GPU, rank0, rank1

```
rank 0: X00,  -,   -,   -
rank 0: X30, X31, X32, X33  
rank 1: X20, X21, X22,  -
rank 1: X10, X11,  -,   -
```

可以进一步转化为, 块计算, 如此以来各个rank的计算量相等, 同时相较示例4, 减少计算量为 5/8

```
rank 0: X00, X30, X31, X32, X33  
rank 1: X20, X21, X22, X10, X11
```
