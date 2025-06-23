# python reduce_op.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    
    '''
    print('-'*100)
    print('reduce op')
    print(master_port)
    
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    print('torch distributed is is_initialized:', 
            dist.is_initialized())
    
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print('Rank ', rank, ' has data ', tensor)
    dist.all_reduce(tensor, dist.ReduceOp.SUM)
    if rank == 0:
        print('reduce SUM :', tensor)

    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    dist.all_reduce(tensor, dist.ReduceOp.SUM) # dist.ReduceOp.AVG
    if rank == 0:
        print('reduce AVG :', tensor / world_size)

    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    dist.all_reduce(tensor, dist.ReduceOp.MAX)
    if rank == 0:
        print('reduce MAX :', tensor)
    
    dist.destroy_process_group()


if __name__ == '__main__':

    # 采用 torch 自带的多线程库来模拟6个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)