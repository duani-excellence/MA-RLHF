#!/usr/bin/env python
# https://pytorch.org/tutorials/intermediate/dist_tuto.html#
# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/ptp.py
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_p2p.py



import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp


def gather(tensor, rank, tensor_list=None, root=0, group=None):
    """
        Sends tensor to root process, which store it in tensor_list.
    """
    if group is None:
        group = dist.group.WORLD
    if rank == root:
        assert(tensor_list is not None)
        dist.gather_recv(tensor_list, tensor, group)
    else:
        dist.gather_send(tensor, root, group)

# def run(rank, size):
#     """ Simple point-to-point communication. """
#     print(dist.get_world_size())
#     tensor = torch.ones(1)
#     tensor_list = [torch.zeros(1) for _ in range(size)]
#     dist.gather(tensor, dst=0, gather_list=tensor_list, group=0)

#     print('Rank ', rank, ' has data ', sum(tensor_list)[0])


# # blocking p2p
# def run(rank, size):
#     print('blocking p2p')
#     tensor = torch.zeros(1)
#     print('rank:',rank)
#     print(tensor)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1) # 同步发送
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)

#     print('Rank ', rank, ' has data ', tensor[0])
#     dist.destroy_process_group()

# non-blocking p2p
# 非阻塞式
def run(rank, size):
    # rank = dist.get_rank()
    print('non-blocking p2p')
    tensor = torch.zeros(1)
    print('rank:',rank)
    print(tensor)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1) # 异步发送
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
    dist.destroy_process_group()




def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)



if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
