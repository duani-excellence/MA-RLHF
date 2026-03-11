import ray
import numpy as np

ray.init()

# 远程函数处理列表
@ray.remote
def process_list(data_list):
    # 对列表进行简单处理
    processed = [x * 2 for x in data_list]
    return processed, sum(processed), len(processed)

# 创建测试数据
test_list = [1, 2, 3, 4, 5]

# 并行处理多个列表
futures = [process_list.remote(test_list) for _ in range(3)]
results = ray.get(futures)

print("原始列表:", test_list)
for i, (processed, total, length) in enumerate(results):
    print(f"结果{i+1}: 处理后的列表={processed}, 总和={total}, 长度={length}")