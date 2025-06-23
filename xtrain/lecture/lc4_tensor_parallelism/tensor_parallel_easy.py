# python tensor_parallel_easy.py
'''
以下代码实现:
1. 标准矩阵乘法 forward @ backward
2. 行并行矩阵乘法 forward @ backward
3. 列并行矩阵乘法 forward @ backward

- 前向算法: DP里的 ZeRO-3 需要 gather 完整参数, 才能矩阵计算, 而TP不需要
- 反向算法: TP 反传时需要 进行逐层进行梯度通信, 而 DP 可以一次性算完 梯度reduce
'''

import torch

# 标准矩阵乘法
x = torch.randn(8, 32)
w = torch.randn(32, 64)
label = torch.randn(8, 64)


print('-'*10, 'standard linear', '-'*10)
y = x @ w
print('y:', y[0,:4])
print('y all:', y)

# 损失
e = (y - label) ** 2
loss = e.sum()
print('loss:', loss)

# 梯度计算
grad_y = 2 * (y - label) / y.numel() # 8 x 64
grad_x = grad_y @ w.t()
grad_w = x.t() @ grad_y
print('grad_x:', grad_x[0,:4])
print('grad_w:', grad_w[0,:4])
# print(y)

# 行并行方法
print('-'*10, 'row-tp', '-'*10)
n = 4
w_row = list(torch.split(w, w.shape[0] // 4, dim = 0)) # [ 8 x 64 ]
x_split = list(torch.split(x, x.shape[1] // 4, dim = 1)) # [ 8 x 8 ]
y = torch.zeros_like(label)
for w_sub, x_sub in zip(w_row, x_split):
    y += x_sub @ w_sub 
print('y:', y[0,:4])
print('y all:', y)

grad_x = [ grad_y @ w_sub.t() for w_sub in w_row]
grad_x = torch.cat(grad_x, dim = 1)

grad_w = [ x_tmp.t() @ grad_y for x_tmp in x_split]
grad_w = torch.cat(grad_w, dim = 0)
print('grad_x:', grad_x[0,:4])
print('grad_w:', grad_w[0,:4])


# 列并行方法
print('-'*10, 'col-tp', '-'*10)
n = 4
# x [8 x 32]
w_col = list(torch.split(w, w.shape[1] // 4, dim = 1)) # [ 32 x 16 ]
# y = torch.zeros_like(label)
# print(x.shape)
y_list = [ x @ w_sub for w_sub in w_col ]
y = torch.cat(y_list, dim = 1)
print('y:', y[0,:4])

grad_y = list(torch.split(grad_y, grad_y.shape[1] // 4, dim = 1))
grad_x = torch.zeros_like(x)

# 注意这个reduce方法, 需要发生必要通信算累积梯度和才能进一步往回传播梯度
for w_sub, grad_y_sub in zip(w_col, grad_y):
    grad_x +=  grad_y_sub @ w_sub.t() # 8 x 16, 16 x 32 

grad_w = [ x.t() @ grad_y_sub for grad_y_sub in grad_y]
grad_w = torch.cat(grad_w, dim = 1)
print('grad_x:', grad_x[0,:4])
print('grad_w:', grad_w[0,:4])

