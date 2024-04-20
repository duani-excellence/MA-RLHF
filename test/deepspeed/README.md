# Deepspeed 实操test

- 目标能够基于Deepspeed接口，实现 GPT2 的多卡全参Pretrained训练
- 基于Deepspeed ji


1. 基础训练

```
deepspeed ./test/deepspeed/test_deepspeed_train.py
deepspeed test/deepspeed/hybrid_engine_test.py --deepspeed_config ./config/hybrid_engine_config.json
```

2. 定义Deepspeed ZeRO3 Linear 层

```
deepspeed ./test/deepspeed/test_deepspeed_linear_stage3.py
```

3. 测试 djieepspeed 基础组件
   - init
   - model
   - data
   - optimizer
   - lr schedulers
4. 测试 deepspeed 算子
   - mat
   - add
   - gelu
   - attention
   - rmsnorm
   - softmax
   - 张量计算
5. 测试 deepspeed 训练
   - forward
   - backward
6. 测试 GPT2 训练
   - 创建模型、数据
   - 测试 ZERO-1/2/3
7. 测试分布式训练功能
   - MOE
   - OFFOLOAD
   - PIPE
   - Megatron
   - 多节点并行
