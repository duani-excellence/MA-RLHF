# python col_parallel_linear.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd


class ColFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        output = input @ w
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input = grad_output @ w.t()
        dist.reduce(grad_input, dist.ReduceOp.SUM) #
        grad_w = input.transpose(2,1) @ grad_output
        return  grad_input, grad_w 
    

class ColParallelLinear(nn.Module):
    # 要注意不同 shape 对应的初始化系数不一致
    def __init__(self, dim_in, dim_out, rank = 0, world_size = 1):
        super(ColParallelLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rank = rank
        self.world_size = world_size
        self.w = nn.Linear(dim_in, dim_out // self.world_size , bias = False) 
        

    def forward(self, x):
        return ColFunction.apply( x, self.w.weight.t() ) 
    

    def all_gather_w(self, ):
        w_gather = [torch.zeros(self.dim_in , self.dim_out // self.world_size) for _ in  range(self.world_size) ] 
        dist.all_gather(w_gather, self.w.weight.t())
        w_gather = torch.concat(w_gather, dim = 1 )
        return w_gather


def run_col(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    实现参数列并行, 输入"数据"并行
    输入梯度dx 需要 reduce
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    N = 4
    dim_in = 64
    dim_out = 512
    label = torch.randn(N , dim_out // world_size) # 4x32   32 x 16
    if rank == 0:
        input = torch.randn(N, dim_in, requires_grad=True)
    else:
        input = torch.zeros(N, dim_in, requires_grad=True)
    dist.broadcast(input, src = 0)

    model = ColParallelLinear(dim_in, dim_out, rank, world_size)
    output_tp_col = model(input) 
    loss_fn = nn.MSELoss()
    loss = loss_fn(output_tp_col, label) 

    loss.backward()
    if rank == 0:
        print('tp-col dx:', input.grad[0,:4], input.grad.shape)
        print('tp-col dw:', model.w.weight.grad.t()[0,:4])

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run_col, args=("127.0.0.1", "12801", 2, ), nprocs=2)
