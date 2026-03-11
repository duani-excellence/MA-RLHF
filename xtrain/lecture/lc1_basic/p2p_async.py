# python p2p_async.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    异步通信, 进程A向B传输10个数据, A可以一次性发出, 而在最终进行阻塞直到B接受完数据
    使用 `isend` 和 `irecv` 来实现, 执行返回一个请求 `req`
    `req.wait()` 返回 `True` 表示完成了一个请求， 
    注意传输者在一个 `for` 循环完成传输, 即使接收者未完成接收, 仍然持续传输
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://'+ master_addr + ':' + master_port,
                            rank=rank, 
                            world_size=world_size)
    

    tensor = torch.zeros(1) 
    buffer = []
    print('rank',rank)
    print(tensor)
    if rank == 0:   # 传输者
        # rank_0 依次发送 0,1,...,2 数据, 即使rank 2未即时接受接收到数据
        for i in range(10):
            tensor = torch.ones(1) * i # sum 0:i
            req = dist.isend(tensor=tensor, dst=2) # 异步发送
            print(rank,'->',tensor)
            buffer.append(req)
    elif rank == 2: # 接收者
        # rank_2 
        for i in range(10):
            tmp_tensor = torch.zeros(1) 
            req = dist.irecv(tensor=tmp_tensor, src=0) # 异步接收
            req.wait() 
            tensor += tmp_tensor
            print(rank,'<-',tmp_tensor)
            print(rank,':', tensor)
    
    if rank == 0:
        for req in buffer:
            req.wait()
    # elif rank != 0:
    #     dist.barrier()
    
    # 如果不加 barrier, 那么rank 0 先执行完就执行 destroy 程序了, 导致其他rank hang住
    dist.barrier() 

    print('Rank ', rank, ' has data ', tensor[0])
    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟2个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)