# python compute-balance.py
'''
Stripe Attention

实现思路为: 在 4 个rank里, 切分 8 个子序列, 并且改变子序列顺序
然后每个rank需要处理两个子序列如, 各个rank的空块情况一致, 计算量相同

************rank0*****************
    K0 K1 | K2 K3 | K4 K5 | K6 K7|
Q0  1     |       |       |      |  3 个空块
Q1  1  1  |       |       |      |
----------+-------+-------+------|
Q2  1  1  | 1  1  | 1  1  | 1    |
Q3  1  1  | 1  1  | 1  1  | 1  1 |
*************rank1****************
Q4  1  1  | 1     |       |      |  2个空块
Q5  1  1  | 1  1  |       |      |
----------+-------+-------+------|
Q6  1  1  | 1  1  | 1     |      |  1个空块
Q7  1  1  | 1  1  | 1  1  |      |
*************rank2****************
'''

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
import math
torch.manual_seed(42)

dim = 512
seq_len = 16
world_size = 4

mask = torch.ones(seq_len, seq_len) 
mask = torch.tril(mask)


X = torch.randn(seq_len, dim)
X_chunk = list(torch.chunk(X.clone(), chunks = world_size * 2, dim = 0)) # 8 块数据

balance_map = [0,7,1,6,2,5,3,4]
re_balance_map = [0,2,4,6,7,5,3,1]

# X input remap
X_chunk_remap = []
for i in balance_map:
    X_chunk_remap.append(X_chunk[i])

# mask remap
mask_tmp = torch.zeros_like(mask)
mask_remap = torch.zeros_like(mask)
seg = seq_len // (world_size * 2)
for i in range(world_size*2):
    idx = balance_map[i]
    mask_tmp[i * seg : (i+1) * seg, :] = mask[idx * seg : (idx+1) * seg, :]
for i in range(world_size*2):
    idx = balance_map[i]
    mask_remap[:, i * seg : (i+1) * seg] = mask_tmp[:, idx * seg : (idx+1) * seg]
print(mask_remap)

# 获取Q, K 序列
Wq = torch.randn(dim, dim)
Wk = torch.randn(dim, dim)

Q_chunk = []
K_chunk = []
for i in range(world_size * 2):
    Q_chunk.append(X_chunk_remap[i] @ Wq)
    K_chunk.append(X_chunk_remap[i] @ Wk)

# mask_remap
mask_blocks = torch.chunk(mask_remap, world_size * 2, dim=0)  
mask_blocks = [torch.chunk(mask_block, world_size * 2, dim=1) for mask_block in mask_blocks]  
print(len(mask_blocks))
print(mask_blocks[0][0].shape)

# 标记空块
is_zeros = torch.zeros(world_size * 2, world_size * 2, dtype = torch.bool)
for i in range(world_size * 2):
    for j in range(world_size * 2):
        is_zero = torch.all(mask_blocks[i][j] == 0).bool()
        is_zeros[i][j] = is_zero
print(is_zeros)

# block-attention
result = torch.zeros(seq_len, seq_len) 
for i in range(world_size * 2): 
    if i % 2 == 0:
        count = 0
        r = i // 2
        print(f'----start rank:{r}----') 
    for j in range(world_size * 2):
        if is_zeros[i][j] != True:
            # S = (Q_chunk[i] @ K_chunk[j].t()) * mask_blocks[i][j] / dim
            S = torch.ones(2,2) * mask_blocks[i][j]
            count += 1
            result[ (i * seg) : ((i+1) * seg), (j * seg): ((j+1) * seg)] = S
    if i % 2:
        print('compute block count:', count)
print(result)


# result_remap = torch.zeros(seq_len, seq_len) 
# for i in range(world_size * 2):
#     idx_i = re_balance_map[i]
#     for j in range(world_size * 2):
#         idx_j = re_balance_map[j]
#         result_remap[idx_i * seg: (idx_i+1) * seg, idx_j * seg: (idx_j+1) * seg] = result[i * seg: (i+1) * seg, j * seg: (j+1) * seg]
# print(result_remap)


# mask remap
result_tmp = torch.zeros_like(result)
result_remap = torch.zeros_like(result)
seg = seq_len // (world_size * 2)
for i in range(world_size*2):
    idx = re_balance_map[i]
    result_tmp[:, i * seg : (i+1) * seg] = result[:, idx * seg : (idx+1) * seg]
for i in range(world_size*2):
    idx = re_balance_map[i]
    result_remap[i * seg : (i+1) * seg, :] = result_tmp[idx * seg : (idx+1) * seg, :]
print(result_remap)
