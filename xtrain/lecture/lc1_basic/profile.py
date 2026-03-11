# python profile.py


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    测试大矩阵耗时
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://'+ master_addr + ':' + master_port,
                            rank=rank, 
                            world_size=world_size)

    tensor = torch.randn(4096, 4096)
    # with torch.profiler():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        dist.all_reduce(tensor)
    
    if rank == 0:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)