# python fun_ring_allreduce.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# import torch.distributed.reduce_op as ReduceOp
import time


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    ring all-reduce 包含两个部分
    1. 先计算各GPU梯度
    2. 再将梯度分送到各GPU中
    实现过程注意防止发生死锁
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    batch = torch.zeros(2 * world_size, dtype=torch.int64) + 1 
    print('Rank ', rank, ' has data ', batch)

    micro_batch = list(torch.split(batch, 2, dim = 0))
    reduce_result = torch.zeros(2, dtype=torch.int64)
    tmp_tensor = torch.zeros(2, dtype=torch.int64)

    # stage 1 : reduce scatter
    for i in range(world_size - 1):
        # print(i, rank)
        cur_idx = (  i + rank ) % world_size
        next_idx = ( i + rank + 1) % world_size
        if rank % world_size == 0:
            dist.send(micro_batch[ cur_idx ], dst = (rank + 1) %  world_size  ) # 发送到右侧 rank
            dist.recv(tmp_tensor, src = (rank - 1) %  world_size  ) # 发送到左侧 rank
        else:
            dist.recv(tmp_tensor, src = (rank - 1) %  world_size  )
            dist.send(micro_batch[ cur_idx ], dst = (rank + 1) %  world_size  )
        micro_batch[next_idx] += tmp_tensor
    dist.barrier()
    if rank == 0 :
        print('-' * 100)
        print('Ring All-Reduce Stage1: Reduce Scatter')
    print('Rank ', rank, ' has data ', micro_batch)
    dist.barrier()
    
    # stage 2 : all gather
    for i in range(world_size - 1):
        cur_idx = (i + rank - 1 ) % world_size # rank0的起点为 world_size - 1
        next_idx = (i + rank) % world_size
        if rank % world_size == 0:
            dist.send(micro_batch[ cur_idx ], dst = (rank + 1) %  world_size  )
            dist.recv(tmp_tensor, src = (rank - 1) %  world_size  )
        else:
            dist.recv(tmp_tensor, src = (rank - 1) %  world_size  )
            dist.send(micro_batch[ cur_idx ], dst = (rank + 1) %  world_size  )
        micro_batch[ next_idx ] = tmp_tensor

    dist.barrier()
    if rank == 0 :
        print('-' * 100)
        print('Ring All-Reduce Stage1: All-Gather')
    print('Rank ', rank, ' has data ', micro_batch)
    dist.barrier()

    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)


