# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_demo.py

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time
# import batch_isend_irecv

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(backend='nccl'):
    dist.init_process_group(backend)
    print(f'backend:{backend}')
    rank = dist.get_rank()
    size = dist.get_world_size()
    pid = os.getpid()
    print(f'current pid: {pid}')
    print(f'Current rank {rank}')
    return rank, size

def cleanup():
    dist.destroy_process_group()

def time_sleep():
    print('-'*10)
    time.sleep(0.5)


# class ToyModel(nn.Module):

#     def __init__(self) -> None:
#         super().__init__()
#         self.layer = nn.Linear(1, 1)

#     def forward(self, x):
#         return self.layer(x)


# class MyDataset(Dataset):

#     def __init__(self):
#         super().__init__()
#         self.data = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index:index + 1]


# ckpt_path = '/tmp/tmp.pth'

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def Asynchronous(rank, world_size):
    output = torch.tensor([rank]).cuda(rank)
    s = torch.cuda.Stream()
    handle = dist.all_reduce(output, async_op=True)
    # Wait ensures the operation is enqueued, but not necessarily complete.
    handle.wait()
    # Using result on non-default stream.
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        output.add_(100)
    if rank == 0:
        # if the explicit call to wait_stream was omitted, the output below will be
        # non-deterministically 1 or 101, depending on whether the allreduce overwrote
        # the value after the add completed.
        print(output)

def p2p(rank, world_size):
    send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
    send_tensor.to(0)

    recv_tensor = torch.randn(2, dtype=torch.float32)
    send_tensor.to(1)

    send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    print(recv_tensor)

def broad_cast(rank, size):
    if dist.get_rank() == 0:
        # Assumes world_size of 3.
        objects = ["foo", 12, {1: 2}] # any picklable object
    else:
        objects = [None, None, None]
    # Assumes backend is not NCCL
    device = torch.device("cpu")
    dist.broadcast_object_list(objects, src=0, device=device)
    print(objects)

def all_reduce(rank, size):
    # All tensors below are of torch.int64 type.
    # We have 2 process groups, 2 ranks.
    device = torch.device(f'cuda:{rank}')
    tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
    print(tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(tensor)

    time_sleep()

    # All tensors below are of torch.cfloat type.
    # We have 2 process groups, 2 ranks.
    tensor = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat, device=device) + 2 * rank * (1+1j)
    print(tensor)
    dist.all_reduce(tensor, op=torch.ReduceOp.SUM)
    print(tensor)

def all_gather(rank, size):
    # All tensors below are of torch.int64 dtype.
    # We have 2 process groups, 2 ranks.
    device = torch.device(f'cuda:{rank}')
    tensor_list = [torch.zeros(2, dtype=torch.int64, device=device) for _ in range(2)]
    print(tensor_list)
    tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
    print(tensor)
    dist.all_gather(tensor_list, tensor)
    time_sleep()
    print(tensor_list)

def all_gather_tensor(rank, size):
    # All tensors below are of torch.int64 dtype and on CUDA devices.
    # We have two ranks.
    device = torch.device(f'cuda:{rank}')
    tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank

    time_sleep()
    print(tensor_in)

    # Output in concatenation form
    tensor_out = torch.zeros(size * 2, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(tensor_out, tensor_in)

    time_sleep()
    print(tensor_out)

    # Output in stack form
    tensor_out2 = torch.zeros(size, 2, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(tensor_out2, tensor_in)

    time_sleep()
    print(tensor_out2)

def all_gather_dict(rank, size):
    # Note: Process group initialization omitted on each rank.
    gather_objects = ["foo", {1: 2}] # any picklable object # 多少个卡就有多少个对象
    output = [None for _ in gather_objects]
    print(output)
    dist.all_gather(output, gather_objects[rank])
    print(output)

def scatter(rank, size):
    tensor_size = 2
    t_ones = torch.ones(tensor_size).to(0)
    t_fives = (torch.ones(tensor_size) * 5).to(0)
    output_tensor = torch.zeros(tensor_size).to(rank)

    print(output_tensor)
    if dist.get_rank() == 0:
        # Assumes world_size of 2.
        # Only tensors, all of which must be the same size.
        scatter_list = [t_ones, t_fives]
    else:
        scatter_list = None

    # scatter_list = torch.tensor(scatter_list).to(rank)
    print(scatter_list)

    dist.scatter(output_tensor, scatter_list, src=0)
    # Rank i gets scatter_list[i]. For example, on rank 1:
    print(output_tensor)

def scatter_tensor(rank, size):
    # All tensors below are of torch.int64 dtype and on CUDA devices.
    # We have two ranks.
    device = torch.device(f'cuda:{rank}')
    tensor_out = torch.zeros(2, dtype=torch.int64, device=device)
    print(tensor_out)
    time_sleep()

    # Input in concatenation form
    tensor_in = torch.arange(size * 2, dtype=torch.int64, device=device)
    print(tensor_in)
    time_sleep()

    dist.reduce_scatter_tensor(tensor_out, tensor_in)
    print(tensor_out)
    time_sleep()

    # Input in stack form
    tensor_in = torch.reshape(tensor_in, (size, 2))
    print(tensor_in)
    time_sleep()

    dist.reduce_scatter_tensor(tensor_out, tensor_in)
    print(tensor_out)
    time_sleep()

def all2all_single(rank, size):
    '''
        ref
        input = torch.arange(4) + rank * 4
        input
        # tensor([0, 1, 2, 3])     # Rank 0
        # tensor([4, 5, 6, 7])     # Rank 1
        # tensor([8, 9, 10, 11])   # Rank 2
        # tensor([12, 13, 14, 15]) # Rank 3
        output = torch.empty([4], dtype=torch.int64)
        dist.all_to_all_single(output, input)
        output
        # tensor([0, 4, 8, 12])    # Rank 0
        # tensor([1, 5, 9, 13])    # Rank 1
        # tensor([2, 6, 10, 14])   # Rank 2
        # tensor([3, 7, 11, 15])   # Rank 3
    '''
    input = (torch.arange(2) + rank * 4).to(rank)
    # input.to(rank)
    print(input)
    time_sleep()

    output = (torch.empty([2], dtype=torch.int64)).to(rank)
    # output.to(rank)
    print(output)
    time_sleep()

    dist.all_to_all_single(output, input)
    time_sleep()
    print(output)

def all2all(rank, size):
    '''
    Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.
    Complex tensors are supported.
    '''
    N = 2
    input = (torch.arange(N) + rank * 4).to(rank)
    input = list(input.chunk(N))
    print(input)
    output = list((torch.empty([N], dtype=torch.int64)).to(rank).chunk(N))
    dist.all_to_all(output, input)
    print(output)

def profiles(rank, size):
    with torch.profiler():
        tensor = torch.randn(20, 10).to(rank)
        print(tensor)
        dist.all_reduce(tensor)

def main():
    rank, size = setup('nccl')
    Asynchronous(rank, size)
    all_reduce(rank, size)
    all_gather(rank, size)
    all_gather_tensor(rank, size)
    scatter(rank, size)
    scatter_tensor(rank, size)
    all2all_single(rank, size)
    all2all(rank, size)


    ## all_gather_dict(rank, size) # failed
    ## profiles(rank, size) # failed
    ## p2p(rank, size)

    # rank, size = setup('gloo')
    # broad_cast(rank, size)


    cleanup()


if __name__ == '__main__':
    main()


# send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
# recv_tensor = torch.randn(2, dtype=torch.float32)
# send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
# recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
# reqs = batch_isend_irecv([send_op, recv_op])
# for req in reqs:
#     req.wait()
# recv_tensor
