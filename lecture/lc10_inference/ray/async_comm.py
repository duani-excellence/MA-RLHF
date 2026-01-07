import ray
import time
import random 
import torch
random.seed(42)
torch.manual_seed(42)


ray.init()

@ray.remote
def fun_mul(x, y):
    for i in range(10):
        b = x @ y
        time.sleep(random.random())
        print(f'[MUL] step:{i}')
    return b


@ray.remote
def fun_add(x, y):
    for i in range(10):
        b = x + y
        time.sleep( random.random() )
        print(f'\t\t[ADD] step:{i}')
    return b


A = torch.randn(2048, 2048)
B = torch.randn(2048, 2048)


print('-'*20, 'example 1: 同步执行任务', '-'*20)
result_mul = ray.get(fun_mul.remote(A,B)) # 同步执行任务
print('>>> Return mul shape: ', result_mul.shape)

# `fun_add` 运行过程中, 可见 `fun_mul` 也在打印
result_add = ray.get(fun_add.remote(A,B)) # 同步执行任务
print('>>> Return add shape: ', result_add.shape)

print('-'*20, 'example 2: 异步执行mul任务', '-'*20)
future = fun_mul.remote(A,B) # 异步执行任务

# `fun_add` 运行过程中, 可见 `fun_mul` 也在打印
result_add = ray.get(fun_add.remote(A,B)) # 同步执行任务
print('>>> Return add shape: ', result_add.shape)

result_mul = ray.get(future) # 阻塞异步任务
print('>>> Return mul shape: ', result_mul.shape)


