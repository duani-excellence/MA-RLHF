# ref: https://github.com/liangan1/Tensor-Parallel-PyTorch/blob/main/test_torch_dtensor_tp.py
# anathor ref: https://zhuanlan.zhihu.com/p/62600N269
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_tp.py
# 1. 创建basic的模型，此时再两卡上各分配526MB显存
# 2. 基于basic的模型，做TP切分，模型再两个卡上各分为 263，263
# 3. 此时观察到两卡显存占用到[526+263~=814,526+263~=814]
# 4. 所做inference正常

import torch
import os
import torch.nn as nn
from copy import deepcopy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, init_device_mesh
import time

NUM_DEVICES = 2

# D = 128   # 特征维度
# HD = 512  # 隐含层特征维度
# N = 64    # 数据条目

# debug
D = 4096   # 特征维度
HD = 9182  # 隐含层特征维度 debug
N = 256    # 数据条目

class MLPModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(D, HD, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(HD, D, device=device)

    def forward(self, x):
        act1 = self.relu(self.net1(x))
        print("act1:", act1.shape)
        return self.net2(act1)

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


device_type = "cuda"

device_mesh = init_device_mesh(device_type, mesh_shape=(2,))

inp_size = [N, D]
# Ensure all tp ranks have same input.
torch.manual_seed(0)

rank = dist.get_rank()

inp = torch.rand(*inp_size, device=rank)
model = MLPModule(rank)


time.sleep(10)
print("Original model:", model)


time.sleep(10)
print('copy ....')
model_tp = deepcopy(model)


time.sleep(10)
print('tp ....')
# Shard module policy
parallelize_plan = {
    "net1": ColwiseParallel(),
    "net2": RowwiseParallel(),
}
model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)

print("After Tensor Parallel model_tp: ", model_tp)

#inference with BF16
with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True):

    time.sleep(10)
    print('original  model inference ....')
    output = model(inp)
    print(output)


    time.sleep(10)
    print('tp  model inference ....')
    print('-'*50)

    output_tp = model_tp(inp)
    print(output_tp)
