# python deepseek-v3.py

"""
only implemented MoE modules
1. basic MoE
2. Auxiliary-Loss-Free Load Balancing
3. Complementary Sequence-Wise Auxiliary Loss
4. why use sigmoid(gate) instead softmax(gate)
"""


import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
torch.manual_seed(42)


class SwiGLU(nn.Module): # expert
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


class DeepSeekV3MoE(nn.Module):
    def __init__(self, dim, expert_nums = 8, top_k = 2, shared_expert_nums = 4):
        super().__init__()

        # Route Experts: 
        self.expert_nums = expert_nums
        self.k = top_k
        self.experts = nn.ModuleList()
        for _ in range(self.expert_nums):
            self.experts.append(SwiGLU(dim))
        self.w_gate = nn.Linear(dim, self.expert_nums)
        self.sigmoid = nn.Sigmoid() # 增加 sigmoid 产生有限的门控值,替换 Softmax

        # Auxiliary-Loss-Free Load Balancing
        # when gradient update, has `gamma` hype-parameter called bias update speed
        self.bias = torch.nn.Parameter(torch.zeros( expert_nums )) 
        
        # Shared Experts: 
        self.shared_expert_nums = shared_expert_nums
        self.shared_experts = nn.ModuleList()
        for _ in range(self.shared_expert_nums):
            self.shared_experts.append(SwiGLU(dim))

        # for load balance
        self.alpha = 0.001


    def forward(self, x):
        y_route, weight, idx = self.forward_route(x) 
        y_shared = self.forward_shared(x) 
        y = x + y_route + y_shared

        load_loss = self.load_balance_sequence_wise(weight, idx)

        return y, load_loss
    
    def forward_route(self, x):
        g = self.w_gate(x)

        # Why use Sigmoid() to instead softmax() ?
        # Mixtral use Softmax(): https://github.com/huggingface/transformers/blob/85f060e9b061d2302b4a6a1f97928463bc6436bd/src/transformers/models/mixtral/modeling_mixtral.py#L112
        # `lecture/lc7_MoE/sigmoid_gate.py`
        g = self.sigmoid(g)
        # g = F.softmax(g, dim = -1) 
        weight, idx = torch.topk(g, k = self.k, dim = -1) 
        # 归一化, 权重和为 1
        weight_norm = weight / (weight.sum(dim=-1, keepdim= True) + 1e-20)

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
                y_result[cur_pos[0], cur_pos[1], :] += expert_results[i] * weight_norm[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)
        # reture basic gate `g` for compute load-balance
        return y_result, g, idx 


    def forward_shared(self, x):
        y = torch.zeros_like(x)
        for i in range(self.shared_expert_nums):
            y += self.shared_experts[i](x) # not gate
        return y

    def load_balance_sequence_wise(self, s, idx):
        '''
        sequence-wise 是为了在 inference 阶段, prefill 时能够均匀分散在各 GPU 中

        这里仅在一层网络里, 一个 batch 的 load balance.
        问题：
            1. 是否是层独立
            2. 是否是所有层上, 统计 load balance
        '''
        Nr = self.expert_nums # n routes expert
        bs, seq_len, dim = s.shape # seq : pad token 要去除，或者增加 mask, 避免计入 loss 里

        l_lab = torch.zeros(1)

        for k in range(bs):

            # Compute fi
            fi = torch.zeros(Nr)
            # pi = torch.zeros(self.expert_nums)

            # seq-count 
            seq_expert_count = torch.zeros(Nr)
            idx_seq = idx[k,:,:]

            for i in range(self.expert_nums):
                seq_expert_count[i] = torch.where(idx_seq == i)[1].numel()
            
            fi = Nr / (self.k * seq_len) * seq_expert_count

            # Compute pi
            s_seq = s[k, :, :]
            si_ = s / s.sum(dim = -1, keepdim = True)
            pi = si_.sum(dim = 0) / seq_len

            l_bal_seq = (fi * pi).sum() / seq_len # seq_len_no_pad or use mask
            l_lab += l_bal_seq

        l_lab = self.alpha * l_lab
        return l_lab


# config
bs = 2
seq_len = 8
dim = 64
expert_nums = 20
expert_shared_nums = 4
topk = 2
x = torch.randn(bs, seq_len, dim)
label = torch.randn(bs, seq_len, dim)

# mode
model = DeepSeekV3MoE(dim, expert_nums=expert_nums, top_k = topk, shared_expert_nums = expert_shared_nums)
print(model)
y, load_loss = model(x)
print(y)
print(load_loss)
