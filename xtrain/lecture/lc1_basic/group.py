# python group.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def group(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    我们先创建一个大组初始化, 再分别创建两个子组
    multi-group
        1. multi-process initial
        2. create 2 group
        3. get group-global rank index
    如果运行命令制定了IP和PORT, 那么是否这个组就是共享这个IP:PORT?
    如果多个node,其ip和port应该如何设置?
    '''
    print('-'*100)
    print('distributed device-group')
    print(master_port)
    
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    print('torch distributed is is_initialized:', 
            dist.is_initialized())
    
    print('current process rank:', dist.get_rank())

    group_0 = dist.new_group(ranks =[0,1,2],
                                backend='gloo')
    group_1 = dist.new_group(ranks =[3,4,5],
                                backend='gloo')
    
    # 只在rank0上打印, 查看分组情况
    if dist.get_rank() == 0:
        print('-'*100)
        print(dist.get_group_rank(group = group_0,
                                global_rank=2 ))
        print(dist.get_global_rank(group = group_0,
                                group_rank=0 ))
        print(dist.get_process_group_ranks(group_0))

        print('-'*100)
        print(dist.get_group_rank(group = group_1,
                                global_rank=5 ))
        print(dist.get_global_rank(group = group_1,
                                group_rank=0 ))
        print(dist.get_process_group_ranks(group_1))
    dist.destroy_process_group()

def node_group(rank, master_addr, master_port, world_size, backend='gloo'):
    '''

    '''
    return


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

if __name__ == '__main__':
    master_port = find_free_port()

    # 采用 torch 自带的多线程库来模拟6个进程执行
    mp.spawn(group, args=("127.0.0.1", master_port, 6, ), nprocs=6)