# by xiaodongguaAIGC
# 将数据提交给远程设备上进行计算， 如 cpu -> mps/cuda

import ray
import torch

ray.init()

# device = 'cuda:0'
device = 'mps' # mac 电脑

# 远程函数处理列表
@ray.remote
def process(data, device):
    from_device = data.device
    data.to(device)
    return data.sum().to(from_device), data.max().to(from_device)

# 创建测试数据
A = torch.randn(10,3,4, device='cpu')

# 并行处理多个列表
futures = [process.remote(A[i], device) for i in range(10)]
results = ray.get(futures)

for i, (sum, max) in enumerate(results):
    print(f"结果{i+1}:  总和={sum}, 最大={max}")
    