# python fun_gather.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# import torch.distributed.reduce_op as ReduceOp
import time


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 gather 函数
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # all gather  tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all gather  tensor')
    tensor_list = [torch.zeros(2,dtype=torch.int64) for _ in range(world_size)]
    print('Rank ', rank, ' has data ', tensor_list)
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print('Rank ', rank, ' has data ', tensor)

    dist.all_gather(tensor_list = tensor_list, 
                    tensor = tensor)
    print('Rank ', rank, ' has data ', tensor_list)

    # all gather into tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all gather [into] tensor')
    tensor_trg_col = torch.zeros(2 * world_size, dtype=torch.int64) # col-wise
    tensor_trg_row = torch.zeros(world_size, 2, dtype=torch.int64) # row-wise
    # tensor_trg_row = torch.zeros(2,world_size, dtype=torch.int64) # col-wise
    print('Rank ', rank, ' has data ', tensor_trg_col)

    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    dist.all_gather_into_tensor(output_tensor = tensor_trg_col, 
                                input_tensor = tensor)
    print('Rank ', rank, '[col] has data ', tensor_trg_col)

    tensor = tensor.unsqueeze(dim = 0)
    dist.all_gather_into_tensor(output_tensor = tensor_trg_row, 
                                input_tensor = tensor)
    print('Rank ', rank, '[row] has data ', tensor_trg_row)
    time.sleep(1)
    
    # all gather object
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all gather object')
    gather_objects = ["foo", 12, {1: 2}, 'xiaodongguaAIGC'] # any picklable object
    output = [None for _ in gather_objects]
    tmp_objects = gather_objects[rank]
    print(f'rank:{rank}', tmp_objects)
    print(f'rank:{rank}', output)
    # time.sleep(1)
    dist.all_gather_object(output, tmp_objects)
    print(f'rank:{rank}', output)


    dist.destroy_process_group()


def p2p_gather(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    手动实现 all gather into tensor
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    group = dist.new_group(ranks = [0,1,2,3])
    
    # all gather tensor
    if rank == 0:
        print('-' * 100)
        print('scratch collective funtion: all-gather into tensor')

    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    tensor_trg = torch.arange(2 * world_size)
    tensor_list = [ torch.arange(2, dtype=torch.int64) for _ in range(world_size)]

    # rank0 收集 list
    if rank == 0:
        tensor_list[0] = tensor
        ranks = dist.get_process_group_ranks(group)
        for r in ranks :
            if r != 0:
                dist.recv(tensor_list[r], src = r)
    else:
        dist.send(tensor, dst = 0)

    if rank == 0:
        print(f'rank:{rank} gather result', tensor_list)
    
    # rank0 合并tensor
    if rank == 0:
        tensor_trg = torch.concat(tensor_list, dim = 0)
        print(f'rank:{rank} gather concat result', tensor_trg)
    
    # 将rank0 结果广播给其他GPU
    dist.broadcast(tensor_trg, src = 0)
    print(f'rank:{rank} all-gather result', tensor_trg)
    
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)

    # 手动实现
    mp.spawn(p2p_gather, args=("127.0.0.1", "12801", 4, ), nprocs=4)