# python fun_all2all_ascyn.py.py 


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
    reqs_isends = []
    reqs_irecvs = []
    for i in range(world_size):
        print(i)
        if i != rank: # 不用避免死锁
            req_isend = dist.isend(input_list[i], dst = i) 
            req_irecv = dist.irecv(output_list[i], src = i)
            reqs_isends.append(req_isend)
            reqs_irecvs.append(req_irecv)
        else:
            output_list[i] = input_list[i]

    # comm-comp overlap
    A = torch.randn(4096, 4096) @ torch.randn(4096, 4096)

    for req in reqs_irecvs:
        req.wait()


    print('Rank ', rank, ' has data ', output_list)
    dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(p2p_all2all, args=("127.0.0.1", "12801", 4, ), nprocs=4)