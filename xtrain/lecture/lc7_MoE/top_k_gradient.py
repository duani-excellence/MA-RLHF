# python top_k_gradient.py

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

# torch autograd
dim = 16
experts_num = 8
top_k = 2
bs = 2
label = torch.randn(bs, top_k)
a = torch.randn(bs, dim, requires_grad=True)
w_gate = nn.Linear(dim, experts_num)
y = w_gate(a)
y.retain_grad()
p = F.softmax(y, dim = -1)
p.retain_grad()
v, idx = torch.topk(p, k = top_k, dim = -1)
print(f'topk idx: {idx}')
loss = ((v - label) ** 2 ) / label.numel()
loss = loss.sum()
loss.backward()
print('pytorch top-k p grad', p.grad)
print('pytorch input_gradient: ' , a.grad[0,:4])
print('pytorch w_gate: ' , w_gate.weight.grad.t()[0,:4])

print('-' * 50)
# hand-write
dv = 2 * (v - label) / label.numel()
dp = torch.zeros(bs, experts_num)
# only backward select-top-k element's gradient
for i in range(bs):
    dp[i, idx[i]] = dv[i,:]
print('hand-write top-k p grad', dp)

dy = torch.zeros_like(y)
for i in range(bs):
    tmp_p = torch.zeros(experts_num)
    # tmp_p[idx[i]] = p[i, idx[i]] 
    tmp_p = p[i,:]
    d_s = torch.diag(tmp_p) - torch.outer(tmp_p, tmp_p)
    dy[i, :] = dp[i, :] @ d_s  
print(dy)

    
d_a = dy @ w_gate.weight
d_w_gate = (a.t() @ dy)
print('hand-write input_gradient: ' , d_a[0,:4])
print('hand-write w_gate: ' , d_w_gate[0,:4])