# by xiaodongguaAIGC
# 实现分布式操作,如 all-reduce
# 数据源在 GPU:0,1,2,3,...
# reduce 操作在 GPU:0 上
# 返回结果 to 各个 GPU 上

import ray
import torch
import time

ray.init()

device = 'mps'  if torch.mps.is_available() else 'cuda:0'
# device = 'cuda:0'

# 远程函数处理列表
@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0) 
def all_reduce_mean(refs):
    
    # 传输到目标计算设备中, 如 cuda:0
    tensors = ray.get(refs)
    tensors = [tensor.to(device) for tensor in tensors]
    
    # 计算
    tensors_cat = torch.cat(tensors, dim=0)
    result = tensors_cat.mean(dim=0)
    
    return result

# 创建测试数据
data_list = [torch.randn(1,3,4, device='cpu') for _ in range(8)] # GPUx8
refs = [ray.put(data) for data in data_list]
results = ray.get(all_reduce_mean.remote(refs)).to('cpu')

print(results)
    
    
time.sleep(3)