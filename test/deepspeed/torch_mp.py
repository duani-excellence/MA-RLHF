# ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#
# run with
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_mp.py

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

def demo_model_parallel(rank, world_size):
    # print(f"Running DDP with model parallel example on rank {rank}.")
    # setup(rank, world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    rank = dist.get_rank()
    print(dist)
    print(f"Start running basic DDP example on rank {rank}.")

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.01)

    epochs = 10
    input_ids = torch.randn(20, 10)
    labels = torch.randn(20, 5).to(dev1)
    print('-'*50)
    print(input_ids)
    print(labels)

    for i in range(epochs):
        optimizer.zero_grad()
        # outputs will be on dev1
        outputs = ddp_mp_model(input_ids)
        loss = loss_fn(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()
    print('haha')

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(n_gpus)
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    world_size = n_gpus//2
    print('word_size = ', world_size)
    run_demo(demo_model_parallel, world_size)
