# ref: https://github.com/liangan1/Tensor-Parallel-PyTorch/blob/main/test_torch_dtensor_tp.py
# anathor ref: https://zhuanlan.zhihu.com/p/626008269


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

NUM_DEVICES = 2

class MLPModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, device=device)

    def forward(self, x):
        act1 = self.relu(self.net1(x))
        print("act1:", act1.shape)
        return self.net2(act1)

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


device_type = "cpu"

device_mesh = init_device_mesh(device_type, mesh_shape=(2,))

inp_size = [8, 10]
# Ensure all tp ranks have same input.
torch.manual_seed(0)
inp = torch.rand(*inp_size, device=device_type)
model = MLPModule(device_type)
print("Original model:", model)
model_tp = deepcopy(model)

# Shard module policy
parallelize_plan = {
    "net1": ColwiseParallel(),
    "net2": RowwiseParallel(),
}
model_tp = parallelize_module(model_tp, device_mesh, parallelize_plan)

print("After Tensor Parallel model_tp: ", model_tp)

#inference with BF16
with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True):
    output = model(inp)
    output_tp = model_tp(inp)
