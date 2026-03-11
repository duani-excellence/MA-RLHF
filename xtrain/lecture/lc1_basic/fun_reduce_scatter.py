# python fun_reduce_scatter.py 

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 reduce scatter 函数
    该函数适用于 zero2, 收集所有的梯度
    并将reduce后的梯度进行切分成部分梯度, 并分配到各GPU自行维护特定的梯度/优化器参数
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # scatter list to tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: reduce scatter ')
    
    tensor_out = torch.zeros(2, dtype=torch.int64)
    tensor_in = torch.arange(world_size * 2, dtype=torch.int64,)
    if rank == 0:
        print(tensor_in)
    dist.reduce_scatter_tensor(tensor_out, 
                               tensor_in, 
                               op = dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor_out)
    time.sleep(1)

    # scatter list to tensor stack-form
    if rank == 0:
        print('-' * 100)
        print('collective funtion: reduce scatter stack-form')
    # Input in stack form
    tensor_in = torch.reshape(tensor_in, (world_size, 2))
    tensor_out = torch.zeros(1, 2, dtype=torch.int64)
    print('Rank ', rank, ' has data ', tensor_in)
    dist.reduce_scatter_tensor(tensor_out, tensor_in)
    print('Rank ', rank, ' has data ', tensor_out)

    dist.destroy_process_group()


def p2p_reduce_scatter(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    手动实现 reduce scatter
    1. 先做一次reduce
    2. 再scatter
    '''
    if rank == 0:
        print('-' * 100)
        print('scratch collective funtion: reduce scatter ')


    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    group = dist.new_group(ranks = [0,1,2,3])
    
    tensor_out = torch.zeros(2, dtype=torch.int64)
    tensor_in = torch.arange(world_size * 2, dtype=torch.int64,)
    tensor_reduce = torch.arange(world_size * 2, dtype=torch.int64,)
    
    # rank0收集到数据
    if rank == 0 : 
        ranks = dist.get_process_group_ranks(group)
        tensor_tmp = torch.zeros(world_size * 2, dtype=torch.int64,)
        for r in ranks:
            if r != 0:
                dist.recv(tensor_tmp, src = r)
                tensor_reduce += tensor_tmp
    else:
        dist.send(tensor_in, dst = 0)

    print('Rank ', rank, ' has data ', tensor_reduce)
    time.sleep(1)
    
    # 先切割再传数据
    if rank == 0 : 
        scatter_list = list(tensor_reduce.split(split_size=2, dim = 0))
        ranks = dist.get_process_group_ranks(group)
        tensor_tmp = torch.zeros(world_size * 2, dtype=torch.int64,)
        for r in ranks:
            if r != 0:
                dist.send(scatter_list[r], dst = r)
            else:
                tensor_out = scatter_list[0]
    else:
        dist.recv(tensor_out, src = 0)

    print('Rank ', rank, ' has data ', tensor_out)
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)

    # 手动实现
    mp.spawn(p2p_reduce_scatter, args=("127.0.0.1", "12801", 4, ), nprocs=4)