# python mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd

from row_parallel_linear import RowParallelLinear, RowFunction
from col_parallel_linear import ColParallelLinear, ColFunction

class mlp(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rank = 0, world_size = 1):
        super(mlp, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rank = rank
        self.world_size = world_size
        self.w1 = ColParallelLinear(dim_in, dim_hidden, self.rank, self.world_size ) 
        self.w2 = RowParallelLinear(dim_hidden, dim_out, self.rank, self.world_size )  
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        h = ColFunction.apply(x, self.w1.w.weight.t())
        h_act = self.ReLU(h)
        print(h_act.shape)
        output = RowFunction.apply(h_act, self.w2.w.weight.t())
        return output


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rank = 0, world_size = 1):
        super(SwiGLU, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rank = rank
        self.world_size = world_size
        self.w1 = ColParallelLinear(dim_in, dim_hidden, self.rank, self.world_size ) 
        self.w_act = ColParallelLinear(dim_in, dim_hidden, self.rank, self.world_size ) 
        self.w2 = RowParallelLinear(dim_hidden, dim_out, self.rank, self.world_size )  
        self.SiLU = nn.SiLU()
    
    def forward(self, x):
        h = ColFunction.apply(x, self.w1.w.weight.t())
        h_act = ColFunction.apply(x, self.w_act.w.weight.t())
        h_act = self.SiLU(h_act) * h
        output = RowFunction.apply(h_act, self.w2.w.weight.t())
        return output
    # backward from scratch : lecture/lc7_MoE/smoe.py


def run_mlp(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    mlp 实现, 先列并行, 再行并行
    X[N, d], W1[d, 4d], W2[4d, d]
    XW1' -> N, d;    W2'[d,d]
    [XW1'W2'] -> reduce([N, d])
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    N = 8
    dim_in = 64
    dim_hidden = dim_in * 4
    dim_out = 512

    if rank == 0:
        x = torch.randn(N, dim_in)
        label = torch.randn(N, dim_out)
    else:
        x = torch.zeros(N, dim_in)
        label = torch.zeros(N, dim_out)
    dist.broadcast(x, src = 0)
    dist.broadcast(label, src = 0)
    dist.barrier()

    model = mlp(dim_in = dim_in, 
                dim_hidden = dim_hidden, 
                dim_out = dim_out,
                rank = rank, 
                world_size = world_size)
    
    y = model(x)
    # dist.all_reduce(y, dist.ReduceOp.SUM)

    loss_fn = nn.MSELoss()
    loss = loss_fn(y, label)
    loss.backward()
    print(loss)

    dist.destroy_process_group()



def run_swiglu(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    llama swiglu
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    N = 8
    dim_in = 64
    dim_hidden = dim_in * 4
    dim_out = 512

    if rank == 0:
        x = torch.randn(N, dim_in)
        label = torch.randn(N, dim_out)
    else:
        x = torch.zeros(N, dim_in)
        label = torch.zeros(N, dim_out)
    dist.broadcast(x, src = 0)
    dist.broadcast(label, src = 0)
    dist.barrier()

    model = SwiGLU(dim_in = dim_in, 
                dim_hidden = dim_hidden, 
                dim_out = dim_out,
                rank = rank, 
                world_size = world_size)
    
    y = model(x)
    # dist.all_reduce(y, dist.ReduceOp.SUM)

    loss_fn = nn.MSELoss()
    loss = loss_fn(y, label)
    loss.backward()
    print(loss)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run_mlp, args=("127.0.0.1", "12801", 2, ), nprocs=2)
    mp.spawn(run_swiglu, args=("127.0.0.1", "12801", 2, ), nprocs=2)
