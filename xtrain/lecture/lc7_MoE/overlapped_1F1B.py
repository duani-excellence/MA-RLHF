# python overlapped_1F1B.py
'''
将 DecoderBlock 设计一个 1F1B 计算通信重叠的操作组合。
有两份数据才能支持 overlap, 故输入为 (x0), (dy1, h1)

常规计算: F0(AF0A)(AB1A)B1, 实际上有 8 个运算动作
DecoderBlock.step_basic:
    - h0 = attn(x0)
    - y0 = expert(h0)      
            -> all2all_disp(h0), 
            -> y0 = mlp(h0),
            -> all2all_combine(y0),
    - dh1 = d_expert(dy1)
            -> all2all_disp(dy1), 
            -> dh1 = mlp(dy1),
            -> all2all_combine(dh1),
    - dx1 = d_attn(dh1)

穿插计算: F0(AB1A)(AF0A)B1, 注意这里并没有将通信时间掩盖
    - h0 = attn(x0)
    - dh1 = d_expert(dy1)
            -> all2all_disp(dy1), 
            -> dh1 = mlp(dy1)
            -> all2all_combine(dh1),
    - y0 = expert(h0)
            -> all2all_disp(h0), 
            -> y0 = mlp(h0)
            -> all2all_combine(y0),
    - dx1 = d_attn(dh1)

重叠计算版本: AF0AB1AF0AB1, 写成AF或AB模式, 注意 A 要异步发送, 放置在计算
    - h0 = attn(x0)     【all2all_disp(dy1)】
    - dh1 = mlp(dy1)    【all2all_disp(h0)】
    - y0 = mlp(h0)      【all2all_combine(dh1)】
    - dx1 = d_attn(dh1) 【all2all_combine(y0)】

实现完成

扩展: 同理在推理阶段, 思考如何重叠 0F1F
'''

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import time
# torch.manual_seed(42)

class OverlappedOp():
    """
    将通信操作分离
    """
    def __init__(self, dim = 64, num_experts = 8, rank = 0, world_size = 1):
    # super(OverlappedOp, self).__init__()
        self.dim = dim
        self.rank = rank 
        self.world_size = world_size
        self.num_experts = num_experts
        
        # all-to-all info
        self.all_to_all_map = []
        self.idx_to_experts = []
        self.idx_to_experts_k = []

        # gate
        self.gate_p = None
        self.weight = None

        # isend 
        self.comm_ops = None

        # for forward
        # self.comm_ops_forward = None
        self.dispatch_x = None # tensor
        self.dispatch_y = None # tensor
        self.combine_y = None # tensor

        # for backward 
        # self.comm_ops_backward = None
        self.dispatch_dy = None # tensor
        self.dispatch_dx = None # tensor
        self.combine_dx = None # tensor

        self.is_forward = 0

    def get_all_to_all_map(self, idx):
        '''
        input:
            idx: batch_token_nums, top-k experts ids
        output:
            all_to_all_map: list[tensor[N]]
            nums_to_experts: list[tensor[token_id_to_expert_i]]
        '''
        N = self.num_experts

        self.idx_to_experts = []
        nums_to_experts = torch.zeros(N, dtype = torch.int32)
        self.idx_to_experts_k = []
        for i in range(N):
            pos = torch.where(idx == i) # seq, top1/top2
            nums_to_experts[i] = pos[0].shape[0]
            self.idx_to_experts.append(pos[0]) # 单卡实际处理的token序列, 被分配到专家i
            self.idx_to_experts_k.append(pos[1])
        self.all_to_all_map = [ torch.zeros_like(nums_to_experts) for _ in range(N) ] 

        dist.all_gather(self.all_to_all_map, nums_to_experts)
        return    

    def dispatch_isend(self, x, ):
        _, dim = x.shape
        N = self.num_experts

        dispatch_x_send_list = [  x[self.idx_to_experts[i],:].clone() for i in range(N) ] # send
        dispatch_x_recv_list = [ torch.zeros(self.all_to_all_map[i][self.rank], dim) for i in range(N) ]

        dispatch_x_recv_list[self.rank] = dispatch_x_send_list[self.rank]

        comm_ops = self.all_to_all_asycn_isend(dispatch_x_send_list, dispatch_x_recv_list)
        return comm_ops, dispatch_x_recv_list

    def combine_isend(self, y, ):
        _, dim = y.shape
        N = self.num_experts

        split_size = [ self.all_to_all_map[i][self.rank].item() for i in range(N) ]

        dispatch_y_send_list = list(torch.split(y, split_size_or_sections = split_size, dim = 0))
        dispatch_y_recv_list = [ torch.zeros(self.all_to_all_map[self.rank][i], dim) for i in range(N) ]

        dispatch_y_recv_list[self.rank] = dispatch_y_send_list[self.rank]

        comm_ops = self.all_to_all_asycn_isend(dispatch_y_send_list, dispatch_y_recv_list)
        return comm_ops, dispatch_y_recv_list

    def all_to_all_asycn_isend(self, send_list, recv_list):
        # 异步发送
        comm_ops = []
        for i in range(self.world_size): # world_size == num_experts
            if i != self.rank:
                if send_list[i].shape[0] != 0:
                    req_isend = dist.isend(send_list[i], dst = i)
                    comm_ops.append(req_isend)
                if recv_list[i].shape[0] != 0:
                    req_irecv = dist.irecv(recv_list[i], src = i)
                    comm_ops.append(req_irecv)
            
        return comm_ops

    # def all_to_all_asycn_irecv(self, comm_ops):
    #     # 异步接收
    #     for op in self.comm_ops:
    #         op.wait()
    #     return 
    
    def wait(self, comm_ops):
        if comm_ops :
            for req in comm_ops:
                req.wait()
            comm_ops = []
        return comm_ops
        
class Expert(nn.Module):
    """

    """
    def __init__(self, dim = 64, num_experts = 8, top_k = 2, rank = 0, world_size = 1):
        super(Expert, self).__init__()
        self.dim = dim
        self.rank = rank 
        self.world_size = world_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.MLP = nn.Linear(dim, dim, bias = False)
        self.w_gate = nn.Linear(dim, num_experts, bias = False)
        self.softmax = nn.Softmax(dim = -1)

        self.ops = [
            OverlappedOp(dim = self.dim, num_experts = self.num_experts, 
                        rank = self.rank, world_size = self.world_size),
            OverlappedOp(dim = self.dim, num_experts = self.num_experts, 
                        rank = self.rank, world_size = self.world_size)
        ]

        self.comm_ops = [None, None]

    # def forward(self, x, phase = 0):
    #     """
    #     前向异步 All2All
    #     """
    #     N = self.num_experts
    #     bs, seq_len, dim = x.shape
    #     x = x.reshape(bs * seq_len, dim) # 序列化

    #     # gate 
    #     gate = self.w_gate(x)
    #     self.ops[phase].gate_p = self.softmax(gate)
    #     self.ops[phase].weight, idx = torch.topk(self.gate_p, k = self.top_k, dim = -1) 
    #     self.ops[phase].get_all_to_all_map(idx)

    #     # 异步发送 dispatch
    #     comm_ops, dispatch_x = self.ops[phase].dispatch_isend(x)
    #     self.ops[phase].dispatch_x = dispatch_x

    #     # 接收数据
    #     self.ops[phase].wait(comm_ops)
        
    #     # 计算操作
    #     y = self.MLP(dispatch_x)
    #     self.ops[phase].dispatch_y = y

    #     # 异步发送 combine
    #     comm_ops, combine_y = self.ops[phase].combine_isend(y)

    #     # 接收数据
    #     self.ops[phase].wait(comm_ops)
    #     self.ops[phase].combine_y = combine_y

    #     # 计算操作
    #     y_moe = torch.zeros_like(x)
    #     for i in range(N):
    #         ei = self.ops[phase].combine_y[i]
    #         pos = self.ops[phase].idx_to_experts[i]
    #         k = self.ops[phase].idx_to_experts_k[i]
    #         gi = self.ops[phase].weight[pos, k]
    #         y_moe[pos] += ei * gi.unsqueeze(-1)
    #     y_moe = y_moe.reshape(bs, seq_len, dim)
    #     return y_moe

    # def backward(self, x, dy):
    #     '''
    #     backward的数据流向类似 forward
    #     除了对expert参数要更新外, 还需要分析 w_gate 的计算
    #     '''
    #     # reverse 
    #     N = self.num_experts
    #     bs, seq_len, dim = x.shape
    #     L = bs * seq_len
    #     x = x.reshape(L, dim)
    #     dy = dy.reshape(L, dim)

    #     # 异步发送 dispatch
    #     comm_ops, dispatch_dy = self.ops[phase].dispatch_isend(dy)
    #     self.ops[phase].dispatch_dy = dispatch_dy

    #     # 接收数据
    #     self.ops[phase].wait(comm_ops)
        
    #     # 计算操作
    #     dispatch_dx = dispatch_dy @ self.MLP.weight
    #     self.MLP.weight.grad = dispatch_dy.t() @ self.ops[phase].dispatch_x # 需要用前向中间变量

    #     # 异步发送 combine
    #     comm_ops, combine_dx = self.ops[phase].combine_isend(dispatch_dx)

    #     # 接收数据
    #     self.ops[phase].wait(comm_ops)
    #     self.ops[phase].combine_dx = combine_dx

    #     # Combine gate-weight-expert
    #     dexpert_x = torch.zeros_like(x)
    #     for i in range(N):
    #         dxi = combine_dx[i]
    #         pos = self.idx_to_experts[i]
    #         k = self.idx_to_experts_k[i]
    #         gi = self.weight[pos, k]
    #         dexpert_x[pos] += dxi * gi.unsqueeze(-1)
    #     dexpert_x = dexpert_x.reshape(bs, seq_len, dim)

    #     # 仅展示all-to-all 算子 ignore gate branch backward
    #     # # d_y_dispatch * d_wgate
    #     # d_gi = torch.zeros(L, N)
    #     # for i in range(N):
    #     #     d_wgate_part =(self.combine_y[i] * self.y[i]).sum(-1)
    #     #     pos = self.idx_to_experts[i]
    #     #     d_gi[pos, i] = d_wgate_part

    #     # d_gate = torch.zeros(L, N)
    #     # for i in range(L):
    #     #     ds = torch.diag(self.gate_p[i,:]) - torch.outer(self.gate_p[i,:], self.gate_p[i,:])
    #     #     d_gate[i,:] = d_gi[i, :] @ ds

    #     # self.w_gate.weight.grad = d_gate.t() @ x
    #     # dgate_x = dgate_x.reshape(bs, seq_len, dim)

    #     # dx = dexpert_x + dgate_x
    #     dx = dexpert_x 
    #     return dx

    def forward_gate(self, x, phase = 0):
        """
        前向算子示例
        """
        N = self.num_experts
        bs, seq_len, dim = x.shape
        x = x.reshape(bs * seq_len, dim) # 序列化

        # gate 
        gate = self.w_gate(x)
        self.ops[phase].gate_p = self.softmax(gate)
        self.ops[phase].weight, idx = torch.topk(self.ops[phase].gate_p, 
                                                k = self.top_k, 
                                                dim = -1) 
        self.ops[phase].get_all_to_all_map(idx)
        return x # reshape x

    def dispatch_isend(self, x, phase = 0):
        comm_ops, dispatch_x = self.ops[phase].dispatch_isend(x)
        self.comm_ops[phase] = comm_ops
        return dispatch_x
    
    def wait(self, phase = 0):
        self.comm_ops[phase] = self.ops[phase].wait(self.comm_ops[phase])

    def combine_isend(self, y, phase = 0):
        comm_ops, combine_y = self.ops[phase].combine_isend(y)
        self.comm_ops[phase] = comm_ops
        return combine_y

    def forward_mlp(self, dispatch_x, phase = 0):
        y = self.MLP(dispatch_x)
        self.ops[phase].dispatch_y = y
    
    def forward_combine_moe(self, x, y, phase = 0):
        # 计算操作
        N = self.num_experts
        y_moe = torch.zeros_like(x)
        for i in range(N):
            ei = self.ops[phase].combine_y[i] # 获取中间变量
            pos = self.ops[phase].idx_to_experts[i]
            k = self.ops[phase].idx_to_experts_k[i]
            gi = self.ops[phase].weight[pos, k]
            y_moe[pos] += ei * gi.unsqueeze(-1)
        # y_moe = y_moe.reshape(bs, seq_len, dim)
        return y_moe

    def backward_mlp(self, dispatch_dy, phase = 0):
        # 计算操作
        dispatch_dx = dispatch_dy @ self.MLP.weight
        self.MLP.weight.grad = dispatch_dy.t() @ self.ops[phase].dispatch_x # 需要用前向中间变量
        self.ops[phase].dispatch_dx = dispatch_dx

    def backward_combine_dx(self, x, dy, phase = 0):
        N = self.num_experts
        bs, seq_len, dim = x.shape
        # Combine gate-weight-expert
        dexpert_x = torch.randn(bs * seq_len, dim )
        for i in range(N):
            dxi = self.ops[phase].combine_dx[i] # 获取中间变量
            pos = self.ops[phase].idx_to_experts[i]
            k = self.ops[phase].idx_to_experts_k[i]
            gi = self.ops[phase].weight[pos, k]
            dexpert_x[pos] += dxi * gi.unsqueeze(-1)
        dexpert_x = dexpert_x.reshape(bs, seq_len, dim)
        return dexpert_x

    def forward_step(self, x, phase = 0):
        """
        Standard Ascyn 1F
        """
        dispatch_x = self.dispatch_isend(x, phase = phase)

        self.wait(phase = phase)
        self.ops[phase].dispatch_x = torch.concat(dispatch_x, dim = 0)
        self.forward_mlp(self.ops[phase].dispatch_x, phase = phase)

        combine_y = self.combine_isend(self.ops[phase].dispatch_y, phase = phase)

        self.wait(phase = phase)
        self.ops[phase].combine_y  = combine_y
        y_moe = self.forward_combine_moe(x, self.ops[phase].combine_y, phase = phase)
        return y_moe

    def backward_step(self, x, dy, phase = 0):

        dispatch_dy = self.dispatch_isend(dy, phase = phase)

        self.wait(phase = phase)
        self.ops[phase].dispatch_dy = torch.concat(dispatch_dy, dim = 0)
        self.backward_mlp(self.ops[phase].dispatch_dy, phase = phase)

        combine_dx = self.combine_isend(self.ops[phase].dispatch_dx, phase = phase)

        self.wait(phase = phase)
        self.ops[phase].combine_dx = combine_dx
        dexpert_x = self.backward_combine_dx(x, self.ops[phase].combine_dx, phase = phase)

        return dexpert_x

class DecoderBlockOverlapped(nn.Module):
    def __init__(self, dim = 64, num_experts = 8, top_k = 2, rank = 0, world_size = 1):
        super(DecoderBlockOverlapped, self).__init__()
        self.dim = dim
        self.rank = rank 
        self.world_size = world_size
        self.attn = nn.Linear(dim, dim, bias = False)
        self.expert = Expert(dim, num_experts, top_k, rank, world_size)
    
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
    
    # def step_basic(self, x0, dy1, x1, h1):
    #     y0 = self.forward(x0)
    #     dx1 = self.backward(dy1)
    #     return y0, dx1
    
    # def step_ascyn(self, x0, dy1, x1, h1):
    #     # 该Block一定要接收两份不同的数据
    #     h0 = self.attn(x0)
    #     y0 = self.expert.forward_step(h0) # 异步 ep

    #     dh1 = self.expert.backward(h1, dy1) # 异步 ep
    #     dx1 = dh1 @ self.attn.weight
    #     self.attn.weight.grad = ( dh1.transpose(1,2) @ x1).sum(0)
    #     return y0, dx1
    
    def prefill(self, x, phase = 1):
        """
        先填充 1B 路径的中间变量, 如 all-2-all map
        """
        # x_randn = torch.randn_like(x)
        bs, seq_len, dim = x.shape
        x_reshape = self.expert.forward_gate(x, phase = 1)
        y_moe = self.expert.forward_step(x_reshape, phase = 1) 
        y_moe = y_moe.reshape(bs, seq_len, dim)
        return y_moe
    
    def step_old(self, x0, dy1, h1):
        """
        computation:    F0-attn, B1-mlp,  F0-mlp,  B1-attn
        communication:  B1-disp, F0-disp, B1-comb, F0-comb

        1. 需要抽象 all2all 算子的实现。
        2. 将 Expert 仅作为计算模块, 通信模块独立实现
        3. 以下为伪代码实现
        """

        # B1-disp, F0_attn, 
        # self.expert.b1_dispatch(dy1)
        # h0 = self.attn(x0)

        # F0-disp, B1-disp-wait, B1-mlp, 
        # self.expert.f0_dispatch(h0)
        # self.expert.b1_dispatch_wait()
        # dh1 = self.expert.backward(dy1)

        # B1-comb, F0-disp-wait, F0-mlp
        # self.expert.b1_combine(dh1)
        # self.expert.f0_dispatch_wait()
        # y0 = self.expert.f0_forward(h0)

        # F0-comb, B1-comb-wait, B1-attn
        # y0 = self.expert.f0_combine(y0)
        # self.expert.b1_combine_wait(dh1)
        # dx1 = self.attn.backward(dh1)

        # y0 = self.expert.f0_combine_wait(y0)
        #return y0, dx1

    def step(self, x0, dy1, x1, h1):
        print(dy1.shape)
        bs, seq_len, dim = dy1.shape
        dy1 = dy1.reshape(bs*seq_len, dim)

        # B1-comm dispatch isend 
        dispatch_dy1 = self.expert.dispatch_isend(dy1, phase = 1)
        # F0-comp-attn
        h0 = self.attn(x0)
        reshape_h0 = self.expert.forward_gate(h0, phase = 0)

        # F0-comm isend
        dispatch_h0 = self.expert.dispatch_isend(reshape_h0, phase = 0)
        # B1-comm recv
        self.expert.wait(phase = 1)
        self.expert.ops[1].dispatch_dy = torch.cat(dispatch_dy1, dim = 0)
        # B1-comp-mlp
        self.expert.backward_mlp(self.expert.ops[1].dispatch_dy, phase = 1) # -> dx

        # B1-comm combine isend
        combine_dx1 = self.expert.combine_isend(self.expert.ops[1].dispatch_dx, phase = 1)
        # F0-comm recv
        self.expert.wait(phase = 0)
        self.expert.ops[0].dispatch_x = torch.cat(dispatch_h0, dim = 0)
        # F0 comp-mlp
        self.expert.forward_mlp(self.expert.ops[0].dispatch_x, phase = 0)

        # F0-comm combine isend
        combine_y0 = self.expert.combine_isend(self.expert.ops[0].dispatch_y, phase = 0)
        # B1-comm recv
        self.expert.wait(phase = 1)
        self.expert.ops[1].combine_dx = combine_dx1
        # B1 comp-mlp
        dexpert_x1 = self.expert.backward_combine_dx(x1, self.expert.ops[1].combine_dx, phase = 1)
        dh1 = dexpert_x1
        dx1_attn = dh1 @ self.attn.weight
        self.attn.weight.grad = ( dh1.transpose(1,2) @ x1).sum(0)

        # F0-comm recv
        self.expert.wait(phase = 0)
        self.expert.ops[0].combine_y = combine_y0
        y0_moe = self.expert.forward_combine_moe(h0.reshape(bs*seq_len, dim), 
                                                self.expert.ops[0].combine_y, 
                                                phase = 0)
        y0_moe = y0_moe.reshape(bs, seq_len, dim)

        return y0_moe, dx1_attn

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    EP的本质实际上是: 一套数据并行系统, 中间网络层存在EP
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    bs = 2
    seq_len = 32
    dim = 64
    expert_nums = world_size
    topk = 2
    x0 = torch.randn(bs, seq_len, dim)
    x1 = torch.randn(bs, seq_len, dim)
    h1 = torch.randn(bs, seq_len, dim)
    dy1 = torch.randn(bs, seq_len, dim)

    # init model
    model = DecoderBlockOverlapped(dim, expert_nums, topk, rank, world_size)
    model.replica_param()
    model.prefill(x1, phase = 1)

    # # basic 
    # # # forward
    # # with torch.no_grad():
    # #     y_moe, h = model(x) # h is attention output

    # # # backward
    # # dy = 2 * (y_moe * label) / label.numel()
    # # model.backward(x, dy, h)

    # # 1F1B basic 
    # # model.step_basic(x0, dy1, x1, h1)

    # # # 1F1B ascyn
    # # model.step_ascyn(x0, dy1, x1, h1)

    # # 1F1B overlapped
    if rank == 0:
        print('-'*100)
        print('start comm-comp-overlapped-1F1B')
        print('-'*100)
    y0, dx1 = model.step(x0, dy1, x1, h1)
    if rank == 0:
        print('result')
        print(y0.shape)
        print(dx1.shape)

    # # reduce gradient: attn, w_gate
    # model.all_reduce_gradient()
    
    # dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)
