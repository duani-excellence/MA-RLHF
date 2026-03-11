# python p2p.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    执行 p2p 同步通信, 将tensor数据进行设备间传输
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    

    print('blocking p2p')
    tensor = torch.zeros(1) 
    print('rank',rank)
    print(tensor)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1) # 同步发送
        tensor -= 1
    elif rank == 1:
        dist.recv(tensor=tensor, src=0) # 同步接收
    print('Rank ', rank, ' has data ', tensor[0])
    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟2个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)