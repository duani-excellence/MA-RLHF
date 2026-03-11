# python fun_scatter.py 

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# import torch.distributed.reduce_op as ReduceOp
import time


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 scatter 函数
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # scatter list to tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: scatter ')
    
    tensor_tmp = torch.zeros(2, dtype=torch.int64)
    print('Rank ', rank, ' has data ', tensor_tmp)
    tensor_list = []
    if rank == 0:
        tensor_total = torch.arange(2 * world_size, dtype=torch.int64) + 1 + 2 * rank
        tensor_list = tensor_total.split(split_size=2, dim = 0) # 返回tuple
        print(tensor_list)

    dist.scatter(tensor = tensor_tmp, 
                 scatter_list = list(tensor_list), # 传入list
                 src=0)
    print('Rank ', rank, ' has data ', tensor_tmp)
    time.sleep(1)
    
    # scatter object
    if rank == 0:
        print('-' * 100)
        print('collective funtion: scatter objects')
    if dist.get_rank() == 0:
        # Assumes world_size of 3.
        objects = ["foo", 12, {1: 2}, "xiaodongguaAIGC"] 
    else:
        objects = [None, None, None, None]
    output_list = [None]
    dist.scatter_object_list(output_list, objects, src=0)
    print('Rank ', rank, ' has data ', output_list)

    dist.destroy_process_group()


def p2p_scatter(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    手动实现 reduce scatter
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    group = dist.new_group(ranks = [0,1,2,3])
    
    # all gather tensor
    if rank == 0:
        print('-' * 100)
        print('scratch collective funtion: scatter')

    tensor_tmp = torch.zeros(2, dtype=torch.int64)
    print('Rank ', rank, ' has data ', tensor_tmp)
    tensor_list = []
    if rank == 0:
        tensor_total = torch.arange(2 * world_size, dtype=torch.int64) + 1 + 2 * rank
        tensor_list = tensor_total.split(split_size=2, dim = 0) # 返回tuple
        print(tensor_list)
        ranks = dist.get_process_group_ranks(group)
        for r in ranks:
            if r != 0:
                dist.send(tensor_list[r], dst = r)
            else:
                tensor_tmp = tensor_list[0]
    else:
        dist.recv(tensor_tmp, src = 0)
    
    print('Rank ', rank, ' has data ', tensor_tmp)
    time.sleep(1)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)

    # 手动实现
    mp.spawn(p2p_scatter, args=("127.0.0.1", "12801", 4, ), nprocs=4)