# python smoe_backward.py
'''
Sparse Mixture-of-Expert

1. SMoE forward 实现
2. SMoE backward 实现难点
    - top-k 如何 backward, `lecture/lc7_MoE/top_k_gradient.py`
    - forward 时需要存储什么中间变量, 用于 backward
        - gate, weight, idx
        - 各个专家的输出数据, expert_results
    - backward 时的 w_gate 如何计算
        - y = sum_i (Ei Gi), 对y求导, 则分别对Ei, Gi 分支进行求导, Gi'过程涉及softmax
3. SMoE load-balance
    - 常见做法是什么？
        - Mixtral, from Switch Transformer
    - 对 backward 的影响是什么?
        - L = L_lm + L_balance, 求导 L' = L'_lm / w_gate + L'_balance / w_gate
        - 增加一条分支 反传到 w_gate(合理)
        - L_balance -> fi -> gi -> gate -> wg
        - backward时, 可以先计算 LM 的 梯度 dw_gate, 再计算 load_balance 梯度并叠加原有梯度 dw_gate += d
    - pad token 对 load-balance 影响？
        - 个人猜测是不计入到 load-balance, 甚至All-to-All都会忽略 pad token
    - deepseek-v3 load-balance 方案
        - DeepSeek-v3: `https://github.com/deepseek-ai/EPLB`
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
        return output, (h, h_act, h_act_up)

    def backward(self, x, dy, h, h_act, h_act_up):
        d_h_act_up = dy @ self.w2.weight
        d_w2 = h_act_up.t() @ dy

        # recompute silu
        h_act_silu = torch.sigmoid(h_act) * h_act
        d_h = d_h_act_up * h_act_silu
        # https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
        d_h_act = h_act_silu * ( 1 - h_act_silu ) * h_act + h_act_silu

        d_h_x =  d_h @ self.w1.weight
        d_h_w1 = x.t() @ d_h 

        d_act_x =  d_h_act @ self.w_act.weight
        d_act_w_act = x.t() @ d_h_act 
        d_x = d_h_x + d_act_x

        self.w1.weight.grad = d_h_w1.t()
        self.w_act.weight.grad = d_act_w_act.t()
        self.w2.weight.grad = d_w2.t()
        return d_x

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

        expert_results = []
        expert_tmp = [] # 存储中间变量, 为了手动计算梯度
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i) 
            # 模拟 dispatch 过程, 即选到专家 i 的 embedding 送到 GPUi
            x_select = x[cur_pos[0], cur_pos[1], :] 
            if x_select.shape[0] == 0: # 专家 i 没有 token 被分配到
                expert_results.append(None)
                expert_tmp.append(None)
            else:
                y, tmp = self.experts[i](x_select)
                expert_results.append(y)
                expert_tmp.append(tmp)
        
        # 模拟 combine 过程
        y_result = torch.zeros_like(x) # 假设 mlp 前向计算前后的维度一致
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i) 
            if expert_results[i] != None:
                y_result[cur_pos[0], cur_pos[1], :] += expert_results[i] * weight[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)
        return y_result, weight, idx, expert_tmp, expert_results
    
    def load_balance_loss(self, weight, idx):
        N = self.expert_nums
        bs, seq_len, _= weight.shape
        total_token = bs * seq_len
        fi = torch.zeros(N)
        pi = torch.zeros(N)
        for i in range(N):
            cur_pos = torch.where(idx == i) 
            fi[i] = len(cur_pos[0]) / total_token
            pi[i] = weight[cur_pos].sum() / N
        loss = N * ( fi * pi ).sum()
        return loss, fi, pi
    
    def backward(self, dy, x, expert_tmp, expert_results, weight, idx):
        '''
        不考虑 load_balance_loss, 影响
        需要求解: dw, dx

        由于 y =  Ei * gi, 对 x 求导
        则 y' = Ei' * gi + Ei * gi', 所以对 x 项的求导需要 专家分支 + 门控分支
        '''
        # d_experts = torch.zeros_like()
        d_x_list = []

        # dispatch
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i) 
            weight_scale = weight[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)
            x_select = x[cur_pos[0], cur_pos[1], :]
            dy_select = dy[cur_pos[0], cur_pos[1], :] * weight_scale
            tmp = expert_tmp[i]
            if tmp != None:
                d_x = self.experts[i].backward(x_select, dy_select, tmp[0], tmp[1], tmp[2]) # 已经完成 dw 更新
                d_x_list.append(d_x)
            else:
                d_x_list.append(None)

        # combine, 专家 dx 的梯度
        dexpert_x = torch.zeros_like(x)
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i) 
            if d_x_list[i] != None:
                dexpert_x[cur_pos[0], cur_pos[1], :] += d_x_list[i] * weight[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)

        # 由于 y = Ei * gi
        # 则 y' = Ei' * gi + Ei * gi'
        bs, seq_len, dim = x.shape
        dg = torch.zeros(bs, seq_len, self.expert_nums)
        dg_list = [ torch.zeros(bs, seq_len, self.expert_nums) for _ in range(self.expert_nums)]
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i) 
            weight_scale = weight[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)
            dy_select = dy[cur_pos[0], cur_pos[1], :] * weight_scale
            if expert_results[i] != None:
                d_gi = dy_select * expert_results[i]
                dg_list[i][cur_pos[0], cur_pos[1], i] = d_gi.sum(-1) # 由于 gate 是个标量, 作用在向量里, 所以反向需要sum 
                dg += dg_list[i]

        # dgate = dsoftmax( gate_p ) @ gi'
        gate = self.w_gate(x)
        gate_p = F.softmax(gate, dim = -1)
        dgate = torch.zeros(bs, seq_len, self.expert_nums)
        for i in range(bs):
            for j in range(seq_len):
                ds = torch.diag( gate_p[i,j,:] ) - torch.outer(gate_p[i,j,:], gate_p[i,j,:])
                dgate[i,j,:] = ds @ dg[i,j,:]

        dgate_x = dgate @ self.w_gate.weight
        d_w_gate = ( dgate.transpose(1,2) @ x).sum(dim = 0)
        self.w_gate.weight.grad = d_w_gate

        # 分支合并
        d_moe_x = dgate_x + dexpert_x

        return d_moe_x, dg_list


    def backward_load_balance(self, x, d_gi, pi):
        # load_balance backward
        bs, seq_len, _ = x.shape

        gate = self.w_gate(x)
        gate_p = F.softmax(gate, dim = -1)
        dgate_list = [torch.zeros(bs, seq_len, self.expert_nums) for i in range(self.expert_nums)]
        for i in range(bs):
            for j in range(seq_len):
                ds = torch.diag( gate_p[i,j,:] ) - torch.outer(gate_p[i,j,:], gate_p[i,j,:])
                for k in range(self.expert_nums):
                    dgate_list[k][i,j,:] = ds @ d_gi[k][i,j,:] # dgi / dgate
        
        total = bs * seq_len
        d_loss_gate = torch.zeros_like(x)
        for i in range(self.expert_nums):
            item1 = self.expert_nums * pi[i] # dL_balance / dfi
            item2 = 1 / total # dfi / dgi
            item3 = dgate_list[i]   # dgi / dgate
            d_loss_gate += item1 * item2 * item3
        
        d_w_gate = (d_loss_gate.transpose(1,2) @ x).sum(dim = 0)
        self.w_gate.weight.grad += d_w_gate # 累加
        return 
        

# config
bs = 2
seq_len = 8
dim = 4
expert_nums = 8
topk = 2
x = torch.randn(bs, seq_len, dim)
label = torch.randn(bs, seq_len, dim)

# mode
model = SMoE(dim, expert_nums=expert_nums, top_k = topk)
y, weight, idx, expert_tmp, expert_results = model(x)
print(y.shape)

# load balance
loss, fi, pi = model.load_balance_loss(weight, idx)
print(loss)

dy = 2 * (y - label) / y.numel()

# step1: lm loss backward
dx, dg_list = model.backward(dy, x, 
                    expert_tmp=expert_tmp, 
                    expert_results=expert_results, 
                    weight= weight, 
                    idx = idx,)
# step2: load balance loss backward
model.backward_load_balance(x, dg_list,  pi)

print(dx.shape)