# python fun_all2all.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time


def run_single_all_to_all(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 all2all 函数
    1. single all-to-all : 输入 tensor
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # single all-to-all tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all-to-all tensor')

    input = torch.arange(4) + rank * 4
    output = torch.empty([4], dtype=torch.int64)
    dist.all_to_all_single(output, input)
    print('Rank ', rank, ' has data ', input)
    print('Rank ', rank, ' has data ', output)

    if rank == 0:
        print('-' * 100)
        print('collective funtion: all-to-all equal op')
    scatter_list = list(input.chunk(world_size))
    gather_list = list(output.chunk(world_size))
    for i in range(world_size):
        # 即rank 0的第 0, 1, 2, 3 数据, 从rank 0, 1, 2, 3的第0号数据而来
        dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)
    print('Rank ', rank, ' has data ', input)
    print('Rank ', rank, ' has data ', output)

    dist.destroy_process_group()


def run_single_all_to_all_tensor(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 single all2all 函数
    灵活处理tensor
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # single all-to-all tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all-to-all tensor')

    input = [torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
            torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int64),
            torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64),
            torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.int64)]
    input_splits = torch.tensor([[2, 2, 1, 1],
                                 [3, 2, 2, 2],
                                 [2, 1, 1, 1],
                                 [2, 2, 2, 1]], dtype=torch.int64)
    output_splits = input_splits.t()
    output = [ torch.zeros(torch.sum(output_splits[i,:]), dtype=torch.int64) 
              for i in range(world_size)]
    print(output)

    dist.all_to_all_single(output = output[rank], 
                           input = input[rank], 
                           output_split_sizes = output_splits[rank,:].tolist(), 
                           input_split_sizes = input_splits[rank,:].tolist())
    print('Rank ', rank, ' has data ', output[rank])


    dist.destroy_process_group()

def run_all_to_all(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 all2all 函数
    all-to-all : 输入 tensor list
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # single all-to-all tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all-to-all list')

    input = torch.arange(4) + rank * 4
    input = list(input.chunk(4))
    output = list(torch.empty([4], dtype=torch.int64).chunk(4))
    dist.all_to_all(output, input)
    print('Rank ', rank, ' has data ', output)

    # if rank == 0:
    #     print('-' * 100)
    #     print('collective funtion: all-to-all equal op')
    # scatter_list = input
    # gather_list = output
    # for i in range(world_size):
    #     dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)
    # print('Rank ', rank, ' has data ', gather_list)

    dist.destroy_process_group()

def run_all_to_all_list(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    使用自带的 all2all 函数
    灵活处理tensor
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    # single all-to-all tensor
    if rank == 0:
        print('-' * 100)
        print('collective funtion: all-to-all list')

    input = [torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
            torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int64),
            torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64),
            torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.int64)]
    input_splits = torch.tensor([[2, 2, 1, 1],
                                 [3, 2, 2, 2],
                                 [2, 1, 1, 1],
                                 [2, 2, 2, 1]], dtype=torch.int64)
    input = input[rank].split(input_splits[rank]) 



    output_splits = input_splits.t()
    output = [ torch.zeros(torch.sum(output_splits[i,:]), dtype=torch.int64) 
              for i in range(world_size)]
    output = output[rank].split(output_splits[rank])
    print(output)

    dist.all_to_all_single(output = output,
                           input = input,)

    print('Rank ', rank, ' has data ', output)
    dist.destroy_process_group()

if __name__ == '__main__':
    '''
    使用自带的 all2all 函数
    1. single all-to-all : 输入 tensor
    2. all-to-all : 输入tensor list
    '''
        
    # # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run_single_all_to_all, args=("127.0.0.1", "12801", 4, ), nprocs=4)
    mp.spawn(run_single_all_to_all_tensor, args=("127.0.0.1", "12801", 4, ), nprocs=4)
    # mp.spawn(run_all_to_all, args=("127.0.0.1", "12801", 4, ), nprocs=4)
    # mp.spawn(run_all_to_all_list, args=("127.0.0.1", "12801", 4, ), nprocs=4)
