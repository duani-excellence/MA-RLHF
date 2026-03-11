# python model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd

from rms_norm import RMSNorm
from lm_head import LanguageModelHead
from decoder import Decoder
from embedding import ParallelEmbedding


class XtrainModel(nn.Module):
    def __init__(self, dim, n_kv_heads, heads, num_blocks, vocab_size, rank = 0, world_size = 1):
        super(XtrainModel, self).__init__()
        self.num_blocks = num_blocks
        self.dim = dim 
        self.n_kv_heads = n_kv_heads
        self.heads = heads
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.rank = rank 
        self.world_size = world_size

        self.embedding = ParallelEmbedding(self.dim, self.vocab_size, self.rank, self.world_size)
        self.decoder = Decoder(self.dim, self.n_kv_heads, self.heads, self.num_blocks, self.rank, self.world_size)
        self.rms_norm = RMSNorm(self.dim, self.rank, self.world_size)
        self.lm_head = LanguageModelHead(self.dim, self.vocab_size, self.rank, self.world_size)

        
    def forward(self, x, y): 
        x = self.embedding(x)
        x = self.decoder(x)
        x = self.rms_norm(x)
        loss, logits, log_prob = self.lm_head(x, y)
        return loss, logits, log_prob 


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
    vocab_size = 512


    input = torch.randint(high = vocab_size, size = [bs, seq_len])
    label = torch.zeros_like(input)
    label[:, :-1] =  input[:, 1:]
    label[:, -1] =  input[:, 0]
    dist.broadcast(input, src = 0)
    dist.broadcast(label, src = 0)

    model = XtrainModel(dim, n_kv_heads, heads, num_blocks, vocab_size,rank, world_size)
    loss,_,_ = model(input, label) 
    loss.backward()
    if rank == 0:
        print(loss)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)
