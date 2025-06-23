# python row_parallel_linear.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd


class RowFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        output = input @ w
        dist.all_reduce(output, dist.ReduceOp.SUM)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input = grad_output @ w.t()
        grad_w = input.transpose(2,1) @ grad_output # 行并行仅维护自身的参数 gradient
        return  grad_input, grad_w 
    

class RowParallelLinear(nn.Module):
    # 要注意不同 shape 对应的初始化系数不一致
    def __init__(self, dim_in, dim_out, rank = 0, world_size = 1):
        super(RowParallelLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rank = rank
        self.world_size = world_size
        self.w = nn.Linear(dim_in // self.world_size, dim_out , bias = False) 
        

    def forward(self, x):
        return RowFunction.apply( x, self.w.weight.t() ) 
    
    def row_partrition_input(self, x):
        N, dim_in = x.shape
        input_tp_row = torch.zeros(N, dim_in //self.world_size, requires_grad=True) # 在特征维度进行切分
        if self.rank == 0:
            scatter_list = list(torch.split(x, dim_in // self.world_size, dim = -1 ))
        else:
            scatter_list = []
        dist.scatter(input_tp_row, scatter_list, src = 0)
        dist.barrier()
        return input_tp_row

    def all_gather_w(self, ):
        w_gather = [torch.zeros(self.dim_in // self.world_size, self.dim_out) for _ in  range(self.world_size) ] 
        dist.all_gather(w_gather, self.w.weight.t())
        w_gather = torch.concat(w_gather, dim = 0 )
        return w_gather




def run_row(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    实现行并行, 参数行并行, 输入列并行
    输出需要 reduce
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    N = 4
    dim_in = 8
    dim_out = 6
    label = torch.randn(N, dim_out)
    dist.broadcast(label, src = 0)
    dist.barrier()

    if rank == 0:
        input = torch.randn(N, dim_in)
    else:
        input = torch.zeros(N, dim_in)

    # 行并行
    # w[`32` x 64] -> w1[8x64],w2[8x64],w3[8x64],w4[8x64]
    model = RowParallelLinear(dim_in, dim_out, rank, world_size)

    # 将输入进行切分
    x = torch.zeros(N, dim_in // world_size, requires_grad=True) # 在特征维度进行切分
    if rank == 0:
        scatter_list = list(input.chunk( world_size, dim = -1 )) 
    else:
        scatter_list = []
    dist.scatter(tensor = x, scatter_list = scatter_list, src = 0)
    dist.barrier()

    output_tp_row = model(x) # 每个gpu输出 8x64, 8x64, 8x64, 8x64
    # dist.all_reduce(output_tp_row, dist.ReduceOp.SUM) 

    loss_fn = nn.MSELoss()
    loss = loss_fn(output_tp_row, label) 

    loss.backward()
    if rank == 0:
        print('tp-row dx:', x.grad[0,:4], x.grad.shape)
        print('tp-row dw:', model.w.weight.grad.t()[0,:4])

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run_row, args=("127.0.0.1", "12801", 2, ), nprocs=2)
