# by xiaodongguaAIGC
# 将数据提交给远程设备上进行计算， 如 cpu -> mps/cuda

import ray
import torch
import time

ray.init()

device = 'cuda:0'

# 远程函数处理列表
@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0) 
def process(data):
    print('remote function data device:', data.device)
    return data.sum(), data.max()

# 创建测试数据
A = torch.randn(10,3,4, device='cpu')

# 并行处理多个列表
futures = [process.remote(A[i]) for i in range(10)]
results = ray.get(futures)

for i, (sum, max) in enumerate(results):
    print(f"结果{i+1}:  总和={sum}, 最大={max}")
    
    
time.sleep(3)