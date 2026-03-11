# python smoe_backward.py
'''
GShard

讨论: 如何设计数据并行版本的 GShard 的 Expert-Parallel 

1. GShard forward:
    - 每个GPU部署一个独立专家, attention层重复分布再各GPU上
    - 输入是数据并行, 那么其All-to-All dispatch/combine 更加复杂
    - 在 All-to-All 前需要传输什么数据
    - 忽略负载均衡实现
2. GShard backward:
    - 同理
3. 实现思路:
    - all-gather 全局 门控信息
    - 实现 gate -> dispatch -> compute -> combine
    - 通信实现: a.阻塞, b.非阻塞
    - 门控-梯度流向

进一步思考: 如何实现 compute-communicate-overlap ?
'''

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
# torch.manual_seed(42)

class Experts(nn.Module):
    def __init__(self, dim = 64, num_experts = 8, top_k = 2, rank = 0, world_size = 1):
        super(Experts, self).__init__()
        self.dim = dim
        self.rank = rank 
        self.world_size = world_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.MLP = nn.Linear(dim, dim, bias = False)
        self.w_gate = nn.Linear(dim, num_experts, bias = False)
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, x):
        N = self.num_experts
        bs, seq_len, dim = x.shape
        x = x.reshape(bs * seq_len, dim) # 序列化

        # gate 
        gate = self.w_gate(x)
        self.gate_p = self.softmax(gate)
        self.weight, idx = torch.topk(self.gate_p, k = self.top_k, dim = -1) # idx
        # self.weight_norm = weight / weight.sum(dim = -1, keepdim = True)
        self.all_to_all_map, self.idx_to_experts, self.idx_to_experts_k = self.get_all_to_all_map(idx)

        # All-to-All dispatch
        # print('----Forward All-to-All Dispatch----')
        self.dispatch_x = self.dispatch(x, self.all_to_all_map, self.idx_to_experts)
        # # self.dispatch_x = 

        # print('----Forward Expert Compute----')
        self.y = self.MLP(self.dispatch_x)
        # print(self.y.shape)

        # # # All-to-All Combine
        # print('----Forward All-to-All Combine----')
        self.combine_y = self.combine(self.y, self.all_to_all_map) # [y0, y1, ..., yn]

        # Combine gate-weight-expert
        # print('----Forward SMoE output----')
        y_moe = torch.zeros_like(x)
        for i in range(N):
            ei = self.combine_y[i]
            pos = self.idx_to_experts[i]
            k = self.idx_to_experts_k[i]
            gi = self.weight[pos, k]
            y_moe[pos] += ei * gi.unsqueeze(-1)

        y_moe = y_moe.reshape(bs, seq_len, dim)
        return y_moe

    def backward(self, x, dy):
        '''
        backward的数据流向类似 forward
        除了对expert参数要更新外, 还需要分析 w_gate 的计算
        '''
        # reverse 
        N = self.num_experts
        bs, seq_len, dim = x.shape
        L = bs * seq_len
        x = x.reshape(L, dim)
        dy = dy.reshape(L, dim)
                
        # dispatch
        dispatch_dy = self.dispatch(dy, self.all_to_all_map, self.idx_to_experts)

        # backward
        dispatch_dx = dispatch_dy @ self.MLP.weight
        self.MLP.weight.grad = dispatch_dy.t() @ self.dispatch_x

        # # combine
        combine_dx = self.combine(dispatch_dx, self.all_to_all_map)

        # Combine gate-weight-expert
        dexpert_x = torch.zeros_like(x)
        for i in range(N):
            dxi = combine_dx[i]
            pos = self.idx_to_experts[i]
            k = self.idx_to_experts_k[i]
            gi = self.weight[pos, k]
            dexpert_x[pos] += dxi * gi.unsqueeze(-1)
        dexpert_x = dexpert_x.reshape(bs, seq_len, dim)

        # # dgate_x
        dgate_x = torch.zeros_like(x)

        # d_y_dispatch * d_wgate
        d_gi = torch.zeros(L, N)
        for i in range(N):
            d_wgate_part =(self.combine_y[i] * self.y[i]).sum(-1)
            pos = self.idx_to_experts[i]
            d_gi[pos, i] = d_wgate_part

        d_gate = torch.zeros(L, N)
        for i in range(L):
            ds = torch.diag(self.gate_p[i,:]) - torch.outer(self.gate_p[i,:], self.gate_p[i,:])
            d_gate[i,:] = d_gi[i, :] @ ds

        self.w_gate.weight.grad = d_gate.t() @ x
        dgate_x = dgate_x.reshape(bs, seq_len, dim)

        dx = dexpert_x + dgate_x
        return dx

    def get_all_to_all_map(self, idx):
        '''
        input:
            idx: batch_token_nums, top-k experts ids
        output:
            all_to_all_map: list[tensor[N]]
            nums_to_experts: list[tensor[token_id_to_expert_i]]
        '''
        N = self.num_experts

        idx_to_experts = []
        nums_to_experts = torch.zeros(N, dtype = torch.int32)
        idx_to_experts_k = []
        for i in range(N):
            pos = torch.where(idx == i) # seq, top1/top2
            nums_to_experts[i] = pos[0].shape[0]
            idx_to_experts.append(pos[0]) # 单卡实际处理的token序列, 被分配到专家i
            idx_to_experts_k.append(pos[1])
        all_to_all_map = [ torch.zeros_like(nums_to_experts) for _ in range(N) ] 

        dist.all_gather(all_to_all_map, nums_to_experts)
        return all_to_all_map, idx_to_experts, idx_to_experts_k

    def dispatch(self, x, a2a_map, idx_to_experts):
        N = self.num_experts

        dispatch_x_send_list = [  x[idx_to_experts[i],:] for i in range(N) ] # send
        dispatch_x_recv_list = [ torch.zeros(a2a_map[i][self.rank], self.dim) for i in range(N) ]

        self.all_to_all(dispatch_x_send_list, dispatch_x_recv_list)
        x_expert = torch.concat(dispatch_x_recv_list, dim = 0)

        return x_expert

    def combine(self, y, a2a_map):
        N = self.num_experts

        split_size = [ a2a_map[i][self.rank].item() for i in range(N) ]

        dispatch_y_send_list = list(torch.split(y, split_size_or_sections = split_size, dim = 0))
        dispatch_y_recv_list = [ torch.zeros(a2a_map[self.rank][i], self.dim) for i in range(N) ]

        self.all_to_all(dispatch_y_send_list, dispatch_y_recv_list)
        
        return dispatch_y_recv_list

    def all_to_all(self, send_list, recv_list):
        for i in range(self.world_size): # world_size == num_experts
            if i != self.rank:
                if  i > self.rank : # 避免死锁
                    dist.send(send_list[i], dst = i) 
                    dist.recv(recv_list[i], src = i)
                else: 
                    dist.recv(recv_list[i], src = i)
                    dist.send(send_list[i], dst = i)
            else:
                recv_list[i] = send_list[i]
        return recv_list
    
    def all_to_all_asycn(self, send_list, recv_list):
        comm_ops = []
        for i in range(self.world_size): # world_size == num_experts
            if i != self.rank:
                comm_ops.append(dist.P2POp(dist.isend, send_list[i], i))
                comm_ops.append(dist.P2POp(dist.irecv, recv_list[i], i))
            else:
                recv_list[i] = send_list[i]
        dist.batch_isend_irecv(comm_ops)
        for op in comm_ops:
            op.wait()
        return recv_list
        

class DecoderBlock(nn.Module):
    def __init__(self, dim = 64, num_experts = 8, top_k = 2, rank = 0, world_size = 1):
        super(DecoderBlock, self).__init__()
        self.dim = dim
        self.rank = rank 
        self.world_size = world_size
        self.attn = nn.Linear(dim, dim, bias = False)
        self.expert = Experts(dim, num_experts, top_k, rank, world_size)
    
    def replica_param(self):
        dist.broadcast(self.attn.weight.clone(), src = 0)
        dist.broadcast(self.expert.w_gate.weight.clone(), src = 0)
    
    def all_reduce_gradient(self):
        dist.all_reduce(self.attn.weight.grad, dist.ReduceOp.SUM)
        dist.all_reduce(self.expert.w_gate.weight.grad, dist.ReduceOp.SUM)
        self.attn.weight.grad /= self.world_size
        self.expert.w_gate.weight.grad /= self.world_size
    
    def forward(self, x):

        h = self.attn(x)
        y = self.expert(h)

        return y, h

    def backward(self, x, dy, h):
        dh = self.expert.backward(h, dy)
        dx = dh @ self.attn.weight
        self.attn.weight.grad = ( dh.transpose(1,2) @ x).sum(0)
        return dx



def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    EP的本质实际上是: 一套数据并行系统, 中间网络层存在EP
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    bs = 2
    seq_len = 8
    dim = 64
    expert_nums = world_size
    topk = 2
    x = torch.randn(bs, seq_len, dim) # 每个 rank 的输入都不一样, 看作是 数据并行
    label = torch.randn(bs, seq_len, dim)

    # init model
    model = DecoderBlock(dim, expert_nums, topk, rank, world_size)
    model.replica_param()

    # forward
    with torch.no_grad():
        y_moe, h = model(x) # h is attention output

    # backward
    dy = 2 * (y_moe * label) / label.numel()
    model.backward(x, dy, h)

    # reduce gradient: attn, w_gate
    model.all_reduce_gradient()
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)
