# python online-softmax.py

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

N = 2
seq_len = 9
block_size = 3
x = torch.randn(N, seq_len)
print('original data:', x)
print('-'*50)


print('torch softmax')
p = F.softmax(x, dim = -1)
print('result:', p)
print('-'*50)

print('torch log-softmax')
log_prob = F.log_softmax(x, dim = -1)
print('result log_prob:', log_prob)
print('result prob:', log_prob.exp())
print('-'*50)


print('hand-write softmax')
l = torch.sum(x.exp(), dim = -1, keepdim=True)
p = x.exp() / l
print('softmax prob:', p)
print('-'*50)

print('hand-write safe-softmax')
m, _ = torch.max(x, dim = -1, keepdim=True)
x_m = x - m
l = torch.sum(x_m.exp(), dim = -1, keepdim=True)
p = x_m.exp() / l
print('safe-softmax prob,', p)
print('-'*50)


print('hand-write online-softmax')
x_blocks = x.chunk(block_size, dim = 1)
m_global = torch.ones( N, 1) * -10000.0
l_global = torch.zeros(N, 1)
for x_block in x_blocks:
    m, _ = torch.max(x_block, dim = -1, keepdim=True)
    m_new = torch.maximum(m, m_global )
    x_m = x_block - m_new
    l = torch.sum(x_m.exp(), dim = -1, keepdim=True)
    l_global = l_global * torch.exp(m_global - m_new) + l
    m_global = m_new
p = (x-m_global).exp() / l_global
print('hand-write online-softmax prob,', p)
print('-'*50)


# print('Log-Sum-Exp example:')
# a = torch.randn(5)
# b = torch.randn(5)
# sum_exp_ab = torch.sum(a) + torch.sum(b)
# print(sum_exp_ab) 