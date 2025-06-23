# python vocab_parallel_embedding.py
'''
    embedding矩阵E是二维的, 维度分别为 词表表大小V, 和词向量维度H, 对应两种切分方法:
        1. 词表上切分: 那么就涉及到一个token序列的embedding要从各个GPU上取的
        2. 词向量切分: 每个token的都要从各个GPU上获取sub-embedding, 实现简单
    上述方法里方法1 存在通信不均衡问题, 即可能99% token 从GPU-0获取. 常在数据并行时后采用
    在backward的过程里, 如果一个序列有多次出现第idx号的token, 要注意该token embedding的梯度求和
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd


class VocabParallelEmbeddingFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w, local_embedding_idx, local_idx, rank, world_size):
        ctx.save_for_backward(w, input)
        ctx.custom_obj = [local_embedding_idx, local_idx, rank, world_size]
        local_v, dim = w.shape
        bs, seq_len = input.shape

        local_embedding = torch.zeros(bs, seq_len, dim)
        local_embedding[local_idx[0],
                        local_idx[1],
                        :] = w[input[local_idx[0],local_idx[1]] - rank * local_v,:]

        local_embedding_list = [ torch.zeros_like(local_embedding) for i in range(world_size)]
        dist.all_gather(local_embedding_list, local_embedding)
        for i in range(world_size):
            if i != rank:
                idx_x, idx_y = torch.where(local_embedding_idx == rank)
                local_embedding[idx_x, idx_y, :] = local_embedding_list[i][idx_x, idx_y, :]
        return local_embedding
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        一个token序列为" [1,3,4,3,5], 对应的gradient为 [de1,de3,de4,de3',de5]
        这里的de3和de3'传回来的梯度不一样, 那么我们应该累加梯度为 de3 = de3 + de3'
        '''
        w, input = ctx.saved_tensors
        local_v, dim = w.shape
        local_embedding_idx, local_idx, rank, world_size = ctx.custom_obj
        # local_input = (input * local_idx).int()
        grad_w = torch.zeros_like(w)
        # bs, seq_len = input.shape
        for i,j in zip(local_idx[0], local_idx[1]):
                grad_w[input[i,j] - local_v * rank ] += grad_output[i,j,:]
        return  input, grad_w, None, None, None, None
    

class VocabParallelEmbedding(nn.Module):
    # 要注意不同 shape 对应的初始化系数不一致
    def __init__(self, dim, vocab_size, rank = 0, world_size = 1):
        super(VocabParallelEmbedding, self).__init__()
        self.dim_in = dim
        self.rank = rank
        self.world_size = world_size
        self.embedding = torch.randn(vocab_size // world_size, dim, requires_grad=True)
        self.seg = vocab_size // world_size
        
    def forward(self, x):
        local_embedding_idx = x // self.seg
        local_idx = list(torch.where(local_embedding_idx == self.rank))
        output = VocabParallelEmbeddingFunction.apply(x, self.embedding, 
                                                      local_embedding_idx, local_idx,
                                                      self.rank, self.world_size)
        return output 



class ParallelEmbeddingFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w, rank, world_size):
        ctx.save_for_backward(w, input)
        ctx.custom_obj = [world_size, rank]
        y = w[input,:]
        y_list = [torch.zeros_like(y) for i in range( world_size )]
        dist.all_gather(y_list, y)
        y = torch.cat(y_list, dim = -1)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        world_size, rank = ctx.custom_obj
        vocab_size, dim = w.shape

        grad_w = torch.zeros_like(w)
        bs, seq_len = input.shape
        for i in range(bs):
            for j in range(seq_len):
                    grad_w[input[i,j],:] += grad_output[i, j, dim * rank: dim * (rank+1)]
        return  input, grad_w, None, None
    

class ParallelEmbedding(nn.Module):
    # 要注意不同 shape 对应的初始化系数不一致
    def __init__(self, dim, vocab_size, rank = 0, world_size = 1):
        super(ParallelEmbedding, self).__init__()
        self.dim_in = dim
        self.rank = rank
        self.world_size = world_size
        self.embedding = torch.randn(vocab_size, dim // world_size, requires_grad=True)
        
    def forward(self, x):
        output = ParallelEmbeddingFunction.apply(x, self.embedding, 
                                                self.rank, self.world_size)
        return output 


def run_vocab_parallel_embedding(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    词表并行, megatron, llama3 等框架实现
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    bs = 4
    seq_len = 8
    dim_in = 64
    vocab_size = 512
    label = torch.randn(bs, seq_len , dim_in) # 4x32   32 x 16
    input = torch.randint( high = vocab_size, size=[bs, seq_len])
    dist.broadcast(input, src = 0)

    model = VocabParallelEmbedding(dim_in, vocab_size, rank, world_size)
    output_tp_col = model(input) 
    loss_fn = nn.MSELoss()
    loss = loss_fn(output_tp_col, label) 

    loss.backward()
    if rank == 0:
        print('embedding dw', model.embedding.grad)
    dist.destroy_process_group()



def run_parallel_embedding(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    嵌入维度并行
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    bs = 4
    seq_len = 8
    dim_in = 64
    vocab_size = 512
    label = torch.randn(bs, seq_len , dim_in) # 4x32   32 x 16
    input = torch.randint( high = vocab_size, size=[bs, seq_len])
    dist.broadcast(input, src = 0)

    model = ParallelEmbedding(dim_in, vocab_size, rank, world_size)
    output_tp_col = model(input) 
    loss_fn = nn.MSELoss()
    loss = loss_fn(output_tp_col, label) 

    loss.backward()
    if rank == 0:
        print('embedding dw', model.embedding.grad)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run_vocab_parallel_embedding, args=("127.0.0.1", "12801", 4, ), nprocs=4)
    mp.spawn(run_parallel_embedding, args=("127.0.0.1", "12801", 4, ), nprocs=4)
