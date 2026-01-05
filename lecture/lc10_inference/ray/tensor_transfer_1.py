# by xiaodongguaAIGC
# 使用共享 cuda 共享对象
import ray
import torch
import time

ray.init()

# tensor_gpu = torch.zeros(10, 10, device='cuda:0')
tensor_gpu = torch.zeros(4, 4, device='mps')
tensor_cpu = torch.zeros(2, 2, device='cpu')

tensor_ref = ray.put(tensor_gpu)  # 存储在共享内存中
tensor_ref_cpu = ray.put(tensor_cpu)  # 存储在共享内存中


result = ray.get(tensor_ref)
print(result.device)
print("通过对象引用传输的张量总和:", result)

time.sleep(2)

# 查看共享对象列表
print(tensor_ref)
print(tensor_ref_cpu)
result = ray._private.internal_api.memory_summary()
print(result)