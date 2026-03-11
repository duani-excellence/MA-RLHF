# python sigmoid_gate.py
# TODO: 当前示例不明显
# 核心观点: softmax 鼓励专家竞争, 加剧win-take-all, 初期gi较大，则会抑制其他专家，导致训练不稳定，
# 对输入尺度敏感, 两个相近的值, softmax 差距大


import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

# # config
# t = 100  # tokens
# d = 1024 # dim
# N = 256  # expert nums
# K = 64   # topk

# config
t = 1  # tokens
d = 1024 # dim
N = 10  # expert nums
K = 2   # topk

# data
w_gate = nn.Linear(d, N, bias = False)
X = torch.randn(t, d, requires_grad=True)
label = torch.randn(t, K)
loss_fn = nn.MSELoss()

# softmax -> topk
def softmax_topk(X, label, temperature = 1.0):
    g = w_gate(X) / temperature
    g.retain_grad()
    g_softmax = F.softmax(g, dim = -1)
    weight, _ = torch.topk(g_softmax, k = K)
    weight_norm = weight / (weight.sum(dim = -1, keepdim=True) + 1e-10)
    loss = loss_fn(weight_norm, label)
    loss.backward()
    return g

# sigmoid -> topk
def sigmoid_topk(X, label, temperature = 1.0):
    g = w_gate(X) / temperature
    g.retain_grad()
    g_sigmoid = F.sigmoid(g)
    weight, _ = torch.topk(g_sigmoid, k = K)
    weight_norm = weight / (weight.sum(dim = -1, keepdim=True) + 1e-10)
    loss = loss_fn(weight_norm, label)
    loss.backward()
    return g

t = 0.1
g = softmax_topk(X, label, t)
print(g.grad)
# g.grad = None

# t = 0.01
g = sigmoid_topk(X, label, t)
print(g.grad)
# X.grad = None
