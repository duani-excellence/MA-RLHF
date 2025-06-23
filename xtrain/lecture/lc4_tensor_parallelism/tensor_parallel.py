# python tensor_parallel.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP

class RowParallelLinear(nn.Module):
    # 要注意不同 shape 对应的初始化系数不一致

    def __init__(self, dim_in, dim_out, rank = 0, world_size = 1):
        super(RowParallelLinear, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.w = nn.Linear(dim_in // self.world_size, dim_out , bias = False) 

    def forward(self, x):
        return self.w( x )
    
class ColParallelLinear(nn.Module):
    def __init__(self, dim_in, dim_out, rank = 0, world_size = 1):
        super(ColParallelLinear, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.w = nn.Linear(dim_in , dim_out // self.world_size , bias = False)

    def forward(self, x):
        # cur_x = x[:, shared_size *  self.rank : shared_size *  (self.rank + 1) ] 
        return self.w( x )
    
def train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        outputs, _ = model(input)
        optimizer.zero_grad()

        loss = loss_fn(outputs, labels)
        loss.backward()     
        optimizer.step()  
        if rank == 0:
            if i % 10 == 0:
                print(loss)

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    1. 实现行并行
    2. 实现列并行
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    N = 8
    dim_in = 128
    dim_out = 256

    if rank == 0:
        input = torch.randn(N, dim_in)
    else:
        input = torch.zeros(N, dim_in)

    # 行并行
    model = RowParallelLinear(dim_in, dim_out, rank, world_size)

    # 备份完整参数, 用于初始化col parallel 初始化参数
    # if rank == 0:
    w_list = [ torch.zeros(dim_in//world_size, dim_out) ] * world_size
    # else:
    #     w_list = []
    dist.all_gather(w_list, model.w.weight.data.t())
    w_gather = torch.concat(w_list, dim = 0)

    input_tp_row = torch.zeros(N, dim_in // world_size) # 在特征维度进行切分
    if rank == 0:
        scatter_list = list(torch.split(input, dim_in // world_size, dim = 1 ))
    else:
        scatter_list = []
    dist.scatter(input_tp_row, scatter_list, src = 0)
    dist.barrier()

    output_tp_row = model(input_tp_row) # 8 x 256 <- 8 x 64, 64 x 256
    print(output_tp_row.shape)
    dist.all_reduce(output_tp_row, dist.ReduceOp.SUM) 
    # output = output_tp_row / world_size
    if rank==0:
        print(output_tp_row)

    # 列并行
    # 对输入在 batch 维度上进行切分, 等同于`数据并行`
    model = ColParallelLinear(dim_in, dim_out, rank, world_size)
    w_list = torch.split(w_gather, dim_out // world_size, dim = 1)
    model.w.weight.data = w_list[rank].t().clone()
    dist.broadcast(input, src = 0)
    dist.barrier()

    output_tp_col = model(input) # 8 x 64 <- 8x128 128 x 64
    print(output_tp_col.shape)
    output = [torch.zeros( N  , dim_out // world_size )] * world_size

    dist.all_gather(output, output_tp_col)

    if rank == 0:
        output_tp_col = torch.concat(output, dim = 1)
        print(output_tp_col)

    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
