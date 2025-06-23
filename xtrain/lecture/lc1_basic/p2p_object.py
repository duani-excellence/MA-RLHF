# python p2p_object.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    执行 p2p 同步通信, 将object数据进行设备间传输
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://'+ master_addr + ':' + master_port,
                            rank=rank, 
                            world_size=world_size)
    objects = []
    if rank == 0:
        objects = ["foo", 12, {1: 2}] # any picklable object
        # dist.send_object_list(objects, dst=1)
        dist.send_object_list(objects, dst=2)
        # dist.send_object_list(objects, dst=3)
    elif rank == 2:
        objects = [None, None, None]
        dist.recv_object_list(objects, src=0)

    print('Rank ', rank, ' has data ', objects)
    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)