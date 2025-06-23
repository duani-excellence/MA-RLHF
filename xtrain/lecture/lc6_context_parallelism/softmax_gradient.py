import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

bs = 2
dim = 6

# torch softmax
softmax = nn.Softmax(dim = -1)
x = torch.randn(bs, dim, requires_grad=True)
p = softmax(x)
p.retain_grad = True
print('x:', x)
print('p:', p)
print('p row-sum:', p.sum(dim = -1, keepdim = True))

# torch.autograd
dy = torch.randn(bs, dim)
torch.autograd.backward(p, dy)
print('torch.autograd: ', x.grad)

# hand-write backward
dx = torch.zeros(bs, dim)
for i in range(bs):
    dp_dx = torch.diag(p[i,:]) - torch.outer(p[i,:], p[i,:])
    dx[i, :] = dy[i, :] @ dp_dx
print('hand-write: ',dx)
