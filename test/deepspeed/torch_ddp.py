# ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#
# run with
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_ddp.py

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import _osx_support

from torch.nn.parallel import DistributedDataParallel as DDP

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(dist)
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    print(ddp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    N = 1000
    epochs = 1000
    input = torch.randn(N, 10).to(device_id)
    labels = torch.randn(N, 5).to(device_id)

    # if rank == 0:
    #     print(input)
    #     print(labels)

    # if rank == 1:
    #     print(input)
    #     print(labels)

    for i in range(epochs):
        # print(i)
        outputs = ddp_model(input)
        loss_fn(outputs, labels).backward() # 仅backward 一次
        optimizer.step()                    # 仅step 一次

    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()
