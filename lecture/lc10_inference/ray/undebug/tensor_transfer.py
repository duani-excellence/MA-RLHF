import ray
import numpy as np

ray.init()

@ray.remote
def tensor_operations(tensor):
    # 简单的张量操作
    mean = np.mean(tensor)
    max_val = np.max(tensor)
    shape = tensor.shape
    return {"mean": mean, "max": max_val, "shape": shape}

# 创建NumPy张量（模拟PyTorch/TensorFlow张量）
tensor1 = np.random.randn(100, 100)  # 100x100随机矩阵
tensor2 = np.ones((50, 50)) * 3.14

# 并行处理张量
future1 = tensor_operations.remote(tensor1)
future2 = tensor_operations.remote(tensor2)

result1 = ray.get(future1)
result2 = ray.get(future2)

print("张量1分析:", result1)
print("张量2分析:", result2)

# 使用对象引用传递大张量
tensor_ref = ray.put(tensor1)  # 存储在共享内存中

@ray.remote
def process_tensor_ref(tensor_ref):
    tensor = ray.get(tensor_ref)  # 从共享内存获取
    print(tensor)
    return 'hh'

result = ray.get(process_tensor_ref.remote(tensor_ref))
print("通过对象引用传输的张量总和:", result)