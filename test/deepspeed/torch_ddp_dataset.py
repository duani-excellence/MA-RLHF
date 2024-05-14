# https://zhuanlan.zhihu.com/p/578610797
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_ddp_dataset.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

def setup():
    dist.init_process_group('nccl')

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


class MyDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.data = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index:index + 1]


ckpt_path = '/tmp/tmp.pth'

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    setup()
    rank = dist.get_rank()
    pid = os.getpid()
    print(f'current pid: {pid}')
    print(f'Current rank {rank}')
    device_id = rank % torch.cuda.device_count()

    dataset = MyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    model = ToyModel().to(device_id)
    ddp_model = DistributedDataParallel(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    if rank == 0:
        torch.save(ddp_model.state_dict(), ckpt_path)

    dist.barrier()

    map_location = {'cuda:0': f'cuda:{device_id}'}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    print(f'rank {rank}: {state_dict}')
    ddp_model.load_state_dict(state_dict)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        for x in dataloader:
            print(f'epoch {epoch}, rank {rank} data: {x}')
            x = x.to(device_id)
            y = ddp_model(x)
            optimizer.zero_grad()
            loss = loss_fn(x, y)
            loss.backward()
            optimizer.step()

            print(f'rank{rank}, loss:{loss}')
            loss_print = reduce_mean(loss, dist.get_world_size())
            print('after reduce:', loss_print)
            print('-'*50)

    cleanup()


if __name__ == '__main__':
    main()
