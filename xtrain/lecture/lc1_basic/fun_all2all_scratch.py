# python fun_all2all_scratch.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time


def p2p_all2all(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    手动实现 all-to-all 函数
    主要存在死锁问题
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    input_list = [ torch.arange(i+1,  dtype=torch.int64 ) + rank * 4 for i in range(world_size)] 
    len_list = [ torch.ones(1, dtype=torch.int64 ) * input.shape[0] for input in input_list]
    trg_list = [torch.zeros(1, dtype=torch.int64)] * world_size

    print('Rank ', rank, ' has data ', input_list)
    print('Rank ', rank, ' has data ', trg_list)
    dist.barrier()

    # 先交换目标tensor维度
    for i in range(world_size):
        if i != rank:
            if  i > rank : # 避免死锁
                dist.send(len_list[i], dst = i) 
                dist.recv(trg_list[i], src = i)
            else: 
                dist.recv(trg_list[i], src = i)
                dist.send(len_list[i], dst = i)
    print('Rank ', rank, ' has data ', trg_list)
    dist.barrier()
    
    # 交换tensor数据
    output_list = [ torch.zeros(i,  dtype=torch.int64 ) for i in trg_list]
    print(output_list)
    for i in range(world_size):
        print(i)
        if i != rank:
            if  i > rank : # 避免死锁
                dist.send(input_list[i], dst = i) 
                dist.recv(output_list[i], src = i)
            else: 
                dist.recv(output_list[i], src = i)
                dist.send(input_list[i], dst = i)
        else:
            output_list[i] = input_list[i]
    print('Rank ', rank, ' has data ', output_list)


    # all2all ascyn version: `lecture/lc7_MoE/fun_all2all_ascyn.py`

    dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(p2p_all2all, args=("127.0.0.1", "12801", 4, ), nprocs=4)