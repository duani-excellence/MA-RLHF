# python fun_reduce.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# import torch.distributed.reduce_op as ReduceOp
import time


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 reduce 函数
    1. 使用 all-reduce, 规约的结果在组内保持一致
    2. 使用 reduce, 规约的结果在目标设备中
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # all reduce tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all reduce tensor')
    tensor = torch.ones(1) * 2 * rank 
    print('Rank ', rank, ' has data ', tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor)

    tensor = torch.ones(1) * 2 * rank 
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    print('Rank ', rank, ' has data ', tensor)

    time.sleep(1)

    # reduce tensor, GLOO not support `reduce`
    if rank == 0:
        print('-' * 100)
        print('collective funtion: reduce tensor')
    tensor = torch.ones(1) * 2 * rank 
    print('Rank ', rank, ' has data ', tensor)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    time.sleep(1)
    print('Rank ', rank, ' has data ', tensor)

    dist.destroy_process_group()


def p2p_reduce(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    手动实现 all_reduce 函数
    1. 以rank0作为参数服务器
    2. p2p 收集其他GPU数据, 并计算sum值
    3. 将sum值广播出去
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    group = dist.new_group(ranks = [0,1,2,3])
    
    # all reudce tensor
    if rank == 0:
        print('-' * 100)
        print('scratch collective funtion: all-reduce tensor')
    tensor = torch.ones(2) * rank * 2
    tensor_sum = torch.ones(2) * rank * 2
    print('Rank ', rank, ' has data ', tensor_sum)
    time.sleep(1)

    # 收集 gpu 数据
    if dist.get_rank() == 0:
        ranks = dist.get_process_group_ranks(group)
        tmp = torch.zeros(2)
        for r in ranks:
            if r != 0:
                dist.recv( tensor=tmp,  src = r)
                tensor_sum += tmp
    elif rank != 0:
         dist.send( tensor=tensor, dst = 0)
        
    if rank == 0:
        print(tensor_sum)
    print('Rank ', rank, ' has data ', tensor_sum)

    # 广播
    dist.broadcast(tensor_sum, src = 0)

    print('Rank ', rank, ' has data ', tensor_sum)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)

    # 手动实现
    mp.spawn(p2p_reduce, args=("127.0.0.1", "12801", 4, ), nprocs=4)