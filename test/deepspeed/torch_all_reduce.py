#!/usr/bin/env python
# https://pytorch.org/tutorials/intermediate/dist_tuto.html#
# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/ptp.py
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_all_reduce.py

import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp


""" All-Reduce example."""
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
    dist.barrier()
    dist.destroy_process_group()

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    # dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    # mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
