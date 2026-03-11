# python col_parallel_linear.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd


class RMSNormFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        eps = 1e-5
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        output = norm * w
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        eps = 1e-5
        # recompute
        x_rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        dw = grad_output * x
        bs, len, dim = x.shape
        I_diag = torch.diag(torch.ones(dim))
        grad_x_rms = torch.zeros_like(x_rms)
        for i in range(bs):
            for j in range(len):
                d_rms = I_diag/x_rms[i,j,:] - (x[i,j,:].outer(x[i,j,:])/x_rms[i,j,0]**3 * dim) * w
                grad_x_rms[i,j,:] = grad_output[i, j, :] @ d_rms
        dist.all_reduce(dw, dist.ReduceOp.SUM)
        return grad_x_rms, dw
    
class RMSNorm(nn.Module):
    # 要注意不同 shape 对应的初始化系数不一致
    def __init__(self, dim, rank = 0, world_size = 1):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.rank = rank
        self.world_size = world_size
        self.gamma =  nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        output = RMSNormFunction.apply(x, self.gamma)
        return output
    

def run_col(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    rms_norm 关键在于 gamma 的全局 reduce
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    N = 4
    seq_len = 10
    dim = 64
    if rank == 0:
        input = torch.randn(N, seq_len, dim, requires_grad=True)
    else:
        input = torch.zeros(N, seq_len, dim, requires_grad=True)
    dist.broadcast(input, src = 0)
    label = torch.randn(N, seq_len, dim)

    model = RMSNorm(dim, rank, world_size)
    rms_norm = model(input) 
    loss_fn = nn.MSELoss()
    loss = loss_fn(rms_norm, label) 

    loss.backward()
    if rank == 0:
        print('rms_norm dx:', input.grad[0,:4], input.grad.shape)
        print('rms_norm dw:', model.gamma.grad)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run_col, args=("127.0.0.1", "12801", 4, ), nprocs=4)
