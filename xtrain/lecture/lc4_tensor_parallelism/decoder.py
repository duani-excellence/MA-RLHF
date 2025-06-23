# python decoder.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd

from row_parallel_linear import RowParallelLinear, RowFunction
from col_parallel_linear import ColParallelLinear, ColFunction
from mlp import SwiGLU
from rms_norm import RMSNorm
from attention import Attention


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_kv_heads, heads, rank = 0, world_size = 1):
        super(DecoderBlock, self).__init__()
        self.dim = dim
        self.rank = rank
        self.world_size = world_size
        self.rms_norm_1 = RMSNorm(dim, rank, world_size)
        self.rms_norm_2 = RMSNorm(dim, rank, world_size)
        self.swiglu = SwiGLU(dim, dim * 4, dim, rank, world_size)
        self.attention = Attention(dim, n_kv_heads, heads, rank, world_size)
    
    def forward(self, x):
        x = self.attention(self.rms_norm_1(x)) + x
        x = self.swiglu(self.rms_norm_2(x)) + x
        return x


class Decoder(nn.Module):
    def __init__(self, dim, n_kv_heads, heads, num_blocks, rank = 0, world_size = 1):
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        self.layers = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(DecoderBlock(dim, n_kv_heads, heads, rank, world_size))
        
    def forward(self, x): # 
        for layer in self.layers:
            x = layer(x)
        return x


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    bs = 16
    seq_len = 32
    dim = 64
    num_blocks = 4
    heads = 8
    n_kv_heads = 2

    label = torch.randn(bs, seq_len, dim) # 4x32   32 x 16
    input = torch.randn(bs, seq_len, dim)
    dist.broadcast(input, src = 0)
    dist.broadcast(label, src = 0)

    model = Decoder(dim, n_kv_heads, heads, num_blocks, rank, world_size)
    output = model(input) 
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, label) 

    loss.backward()
    if rank == 0:
        print(loss)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)
