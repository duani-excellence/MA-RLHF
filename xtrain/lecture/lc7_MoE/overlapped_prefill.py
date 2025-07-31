# python overlapped_prefill.py
'''
由于数据是 2-micro-batch, 简称 F0F1
将 DecoderBlock 设计一个 F0F1 计算通信重叠的操作组合。
有两份数据才能支持 overlap, 故输入为 (x0), (x1)

常规计算: 
DecoderBlock.step_basic:
    - h = attn(x)
    - y = expert(h)      
            -> all2all_disp(h), 
            -> y = mlp(h),
            -> all2all_combine(y),

DecoderBlock 2 micro-batch:
    - h0 = attn(x0)
    - y0 = expert(h0)      
            -> all2all_disp(h0), 
            -> y0 = mlp(h0),
            -> all2all_combine(y0),
    - h1 = attn(x1)
    - y1 = expert(h1)      
            -> all2all_disp(h1), 
            -> y1 = mlp(h1),
            -> all2all_combine(y1),

DecoderBlock 2 micro-batch (new):
    - h0 = attn(x0)
    - h1 = attn(x1)
    - y0 = expert(h0)      
            -> all2all_disp(h0), 
            -> y0 = mlp(h0),
            -> all2all_combine(y0),
    - y1 = expert(h1)      
            -> all2all_disp(h1), 
            -> y1 = mlp(h1),
            -> all2all_combine(y1),

重叠计算版本: AF0AB1AF0AB1, 写成AF或AB模式, 注意 A 要异步发送, 放置在计算
    - [all2all_comb(x1)] h0 = attn(x0)  
    - [all2all_disp(h0)] h1 = attn(x1)  
    - [all2all_disp(h1)] y0 = mlp(h0) 
    - [all2all_comb[y0]] y1 = mlp(h1) 
    - [all2all_comb(y1)] 
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

    def forward_prefill(self, x, phase = 0):
        """
        Standard Ascyn 1F
        """
        dispatch_x = self.dispatch_isend(x, phase = phase)

        self.wait(phase = phase)
        self.ops[phase].dispatch_x = torch.concat(dispatch_x, dim = 0)
        self.forward_mlp(self.ops[phase].dispatch_x, phase = phase)

        # combine_y = self.combine_isend(self.ops[phase].dispatch_y, phase = phase)

        # self.wait(phase = phase)
        # self.ops[phase].combine_y  = combine_y
        # y_moe = self.forward_combine_moe(x, self.ops[phase].combine_y, phase = phase)
        return 


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


    def prefill(self, x, phase = 1):
        """
        先填充 1F 路径的中间变量, 如 all-2-all + mlp, but not combine
        """
        x_reshape = self.expert.forward_gate(x, phase = 1)
        self.expert.forward_prefill(x_reshape, phase = 1) 
        return 
    
    def step_old(self, x0, x1):
        """
        computation:    F0-attn, F1-attn,  F0-mlp,  F1-attn
        communication:  F1-comb, F0-disp,  F1-disp, F0-comb F1-comb

        1. 需要抽象 all2all 算子的实现。
        2. 将 Expert 仅作为计算模块, 通信模块独立实现
        3. 以下为伪代码实现
        """

        # F1-combine isend, F0-attn
        # self.expert.f1_combine(x1)
        # h0 = self.attn(x0)

        # F0-disp, F1-combine-wait, F1-attn
        # self.expert.f0_dispatch(h0)
        # self.expert.f1_combine_wait(x1)
        # h1 = self.attn(x1)

        # F1-disp, F0-dispatch-wait, F0-mlp
        # self.expert.f1_dispatch(h1)
        # self.expert.f0_dispatch_wait(h0)
        # y0 = self.mlp(h0)

        # F0-comb, F0-combine-wait, F1-mlp
        # self.expert.f0_combine(h0)         # micro-0 finish
        # self.expert.f1_dispatch_wait(h1)
        # y1 = self.mlp(h1)

        # F1-comb
        # self.expert.f1_combine_wait(y1)    # micro-1 finish


    def step(self, x0, x1_,):
        bs, seq_len, dim = x0.shape

        # 上一层数据 x1 刚计算完 MLP, 待 combine
        self.prefill(x1_, phase = 1)
        combine_y1 = self.expert.combine_isend(self.expert.ops[1].dispatch_y, phase = 1)
        # F0-attn-comp, F0-attn
        h0 = self.attn(x0) # comp
        reshape_h0 = self.expert.forward_gate(h0, phase = 0)
        

        # dispatch-F0(h0), combine-F1-wait(x1), F1-attn
        dispatch_h0 = self.expert.dispatch_isend(reshape_h0, phase = 0)

        self.expert.wait(phase = 1)
        self.expert.ops[1].combine_y = torch.cat(combine_y1, dim = 0)
        x1 = self.expert.forward_combine_moe(x1_.reshape(bs * seq_len, dim), 
                                             self.expert.ops[1].combine_y, 
                                             phase = 1)
        x1 = x1.reshape(bs, seq_len, dim)
        h1 = self.attn(x1) # comp
        reshape_h1 = self.expert.forward_gate(h1, phase = 1)

        # dispatch-F1(h1), dispatch-F0-wait(), F0-mlp
        dispatch_h1 = self.expert.dispatch_isend(reshape_h1, phase = 1)

        self.expert.wait(phase = 0)
        self.expert.ops[0].dispatch_x = torch.cat(dispatch_h0, dim = 0)
        
        self.expert.forward_mlp(self.expert.ops[0].dispatch_x, phase = 0) # mlp


        # combine-F0(y0), dispatch-F1-wait(), F1-mlp
        combine_y0 = self.expert.combine_isend(self.expert.ops[0].dispatch_y, phase = 0)


        self.expert.wait(phase = 1)

        # for i in range(len(dispatch_h1)):
        #     print(dispatch_h1[0].shape)

        self.expert.ops[1].dispatch_x = torch.cat(dispatch_h1, dim = 0)
        self.expert.forward_mlp(self.expert.ops[1].dispatch_x, phase = 1) # mlp


        # combine-F1(y1), combine-F0-wait()
        combine_y1 = self.expert.combine_isend(self.expert.ops[1].dispatch_y, phase = 1)

        self.expert.wait(phase = 0)
        # process final moe result
        self.expert.ops[0].combine_y = torch.cat(combine_y0, dim = 0)
        y0_moe = self.expert.forward_combine_moe(h0.reshape(bs*seq_len, dim), 
                                                self.expert.ops[0].combine_y, 
                                                phase = 0) # 仅合并数据，不计算
        
        # combine-F1-wait()
        self.expert.wait(phase = 1)
        # process final moe result
        self.expert.ops[1].combine_y = torch.cat(combine_y1, dim = 0)
        y1_moe = self.expert.forward_combine_moe(h0.reshape(bs*seq_len, dim), 
                                                self.expert.ops[1].combine_y, 
                                                phase = 1) # 仅合并数据，不计算
        
        y0_moe = y0_moe.reshape(bs, seq_len, dim)
        y1_moe = y1_moe.reshape(bs, seq_len, dim)

        return y0_moe, y1_moe


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
    x0 = torch.randn(bs, seq_len, dim) # micro-1
    x1 = torch.randn(bs, seq_len, dim) # micro-2

    # init model
    model = DecoderBlockOverlapped(dim, expert_nums, topk, rank, world_size)
    model.replica_param()
    # model.prefill(x1, phase = 1)

    # F0F1 overlapped
    if rank == 0:
        print('-'*100)
        print('start comm-comp-overlapped-1F1B')
        print('-'*100)
    y0, y1 = model.step(x0, x1)
    if rank == 0:
        print('result')
        print(y0.shape)
        print(y1.shape)

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)
