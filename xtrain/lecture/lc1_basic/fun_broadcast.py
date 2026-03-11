# python fun_broadcast.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 broadcast 函数, 进行广播
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # 广播 list
    if rank == 0:
        print('-' * 100)
        print('collective funtion: broadcast list')
    if dist.get_rank() == 0:
        objects = ["foo", 12, {1: 2}] 
    else:
        objects = [None, None, None]
    
    print('Rank ', rank, ' has data ', objects)
    dist.broadcast_object_list(objects, src=0)
    print('Rank ', rank, ' has data ', objects)

    time.sleep(1)

    # 广播 tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: broadcast tensor')
    tensor = torch.zeros(2) 
    if dist.get_rank() == 0:
        tensor = tensor + 100

    print('Rank ', rank, ' has data ', tensor)
    dist.broadcast(tensor, src=0)
    print('Rank ', rank, ' has data ', tensor)

    time.sleep(1)

    # 组广播 tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: broadcast tensor in group')
    tensor = torch.ones(2) * rank * 2
    print('Rank ', rank, ' has data ', tensor)
    group = dist.new_group(ranks = [0,1])
    dist.broadcast(tensor, src=1, group = group)
    print('Rank ', rank, ' has data ', tensor)

    dist.destroy_process_group()


def p2p_broad_cast(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    手动实现 broadcast 函数
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    group = dist.new_group(ranks = [0,1,2,3])
    
    # 广播 tensor
    if rank == 0:
        print('-' * 100)
        print('scratch collective funtion: broadcast tensor')
    tensor = torch.zeros(2) 
    if dist.get_rank() == 0:
        tensor = tensor + 100
        ranks = dist.get_process_group_ranks(group)
        print(ranks)
        for r in ranks:
            if r != 0:
                dist.send( tensor=tensor, dst = r)
    else:
        dist.recv( tensor=tensor, src = 0)

    print('Rank ', rank, ' has data ', tensor)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)

    # 手动实现
    mp.spawn(p2p_broad_cast, args=("127.0.0.1", "12801", 4, ), nprocs=4)