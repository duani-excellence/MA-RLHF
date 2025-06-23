# python smoe_forward.py
'''
Sparse Mixture-of-Expert

1. SMoE forward 实现
'''

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
torch.manual_seed(42)

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super(SwiGLU, self).__init__()
        self.dim_in = dim
        self.dim_out = self.dim_in * 4 # original is 8/3
        self.w1 = nn.Linear(self.dim_in, self.dim_out , bias = False)
        self.w_act = nn.Linear(self.dim_in, self.dim_out, bias = False) 
        self.w2 = nn.Linear(self.dim_out, self.dim_in, bias = False)  
        self.SiLU = nn.SiLU()
    
    def forward(self, x):
        h = self.w1(x)
        h_act = self.w_act(x)
        h_act_up = self.SiLU(h_act) * h
        output = self.w2(h_act_up)
        return output


class SMoE(nn.Module):
    def __init__(self, dim, expert_nums = 8, top_k = 2):
        super(SMoE, self).__init__()
        self.expert_nums = expert_nums
        self.k = top_k
        self.experts = nn.ModuleList()
        for _ in range(self.expert_nums):
            self.experts.append(SwiGLU(dim))
        self.w_gate = nn.Linear(dim, self.expert_nums)

    def forward(self, x):
        '''
        x: [bs, seq_len, dim]
        根据 gate 选择 top-k 专家
        '''
        g = self.w_gate(x)
        g = F.softmax(g, dim = -1)

        # topk example
        # input shape [1,3,4]
        # output: shape [1,3,2], last dim is topk expert ids
        weight, idx = torch.topk(g, k = self.k, dim = -1) 
        # weight = weight / weight.sum(dim = -1)

        expert_results = []
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i) 
            # 模拟 dispatch 过程, 即选到专家 i 的 embedding 送到 GPUi
            x_select = x[cur_pos[0], cur_pos[1], :] 
            if x_select.shape[0] == 0: # 专家 i 没有 token 被分配到
                expert_results.append(None)
            else:
                y = self.experts[i](x_select)
                expert_results.append(y)
        
        # 模拟 combine 过程
        y_result = torch.zeros_like(x) # 假设 mlp 前向计算前后的维度一致
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i) 
            if expert_results[i] != None:
                y_result[cur_pos[0], cur_pos[1], :] += expert_results[i] * weight[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)
        return y_result, weight, idx
    
    def load_balance_loss(self, weight, idx):
        '''
        layer-level load balance loss
        https://zhuanlan.zhihu.com/p/680361287
        follow Mixtral implementation
        weight 的来源是 x @ w_gate, 那么load balance loss是否会增加一条路径反传梯度?
        '''
        N = self.expert_nums
        bs, seq_len, _= weight.shape
        total = bs * seq_len
        count = torch.zeros(N)
        pi_sum = torch.zeros(N)
        for i in range(N):
            cur_pos = torch.where(idx == i) 
            count[i] = len(cur_pos[0]) / total
            pi_sum[i] = weight[cur_pos].sum()
        loss = N * ( count / N * pi_sum / N).sum()
        return loss    

        
# config
bs = 2
seq_len = 8
dim = 64
expert_nums = 20
topk = 2
x = torch.randn(bs, seq_len, dim)
label = torch.randn(bs, seq_len, dim)

# mode
model = SMoE(dim, expert_nums=expert_nums, top_k = topk)
y, weight, idx = model(x)
print(y.shape)

# load balance
loss = model.load_balance_loss(weight, idx)
print(loss)
