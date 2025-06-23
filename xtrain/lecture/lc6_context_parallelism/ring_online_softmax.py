# python ring_online_softmax.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
torch.manual_seed(42)

def safe_softmax_incremental(x, m_global, l_global):
    m, _ = torch.max(x, dim = -1, keepdim=True)
    m_new = torch.maximum(m, m_global)
    l = torch.sum( torch.exp(x-m_new), dim = -1, keepdim=True )
    l_new = l_global * torch.exp(m_global - m_new) + l
    p = (x - m_new).exp() / l_new # ignore softmax output
    return p, m_new, l_new

def safe_softmax(x):
    m, _ = torch.max(x, dim = -1, keepdim=True)
    l = torch.sum( torch.exp(x-m), dim = -1, keepdim=True )
    p = (x - m).exp() / l
    return p, m, l

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    实现增量式的online softmax
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # batch = torch.zeros(2 * world_size, dtype=torch.int64) + 1 
    # print('Rank ', rank, ' has data ', batch)

    bs = 1
    seq_len = 16


    x_local = torch.randn(bs , seq_len // world_size)
    m_global = torch.ones(bs, 1) * -100000.0 
    l_global = torch.zeros(bs, 1)
    x = torch.randn(bs, seq_len)
    if rank == 0:
        '''
        先在各个rank上, 分配列数据:
            rank1       rank2
        | A11, A12,| A13, A14  | 
        | A21, A22,| A23, A24  |
        | A31, A32,| A33, A34  | 
        | A41, A42,| A43, A44  |
        '''
        # print(F.softmax(x , dim = -1))
        x_list = list(x.chunk(world_size, dim = 1))
    else:
        x_list = [] * world_size
    dist.scatter(x_local, x_list, src = 0)
    dist.barrier()

    if rank != 0:
        dist.recv(m_global, src = rank - 1)
        dist.recv(l_global, src = rank - 1)
    _, m_global, l_global = safe_softmax_incremental(x_local, m_global= m_global, l_global= l_global )
    if rank != world_size - 1:
        dist.send(m_global, dst = rank + 1)
        dist.send(l_global, dst = rank + 1)
    dist.barrier()

    if rank == 0:
        print('------------------ring online softmax----------------')
    dist.broadcast(m_global, src = world_size - 1)
    dist.broadcast(l_global, src = world_size - 1)
    local_online_softmax = (x_local - m_global).exp() / l_global
    print(f'{rank}: result: {local_online_softmax}' )

    if rank == 0:
        print('------------------torch softmax----------------')
        p = F.softmax(x , dim = -1)
        # print('torch softmax:', p.chunk(world_size, dim = -1))
        p_list = p.chunk(world_size, dim = -1)
        for r, p in enumerate(p_list):
            print(f'{r}: result: {p}' )


    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)


