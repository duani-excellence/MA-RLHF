# ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#
# run with
# torchrun  distributed_dataset.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import sleep

class MyDataset(Dataset):

    def __init__(self, N, dim, classes):
        super().__init__()
        self.data = torch.randn(N, dim, dtype=torch.float32)
        self.labels = torch.randn(N, classes, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {"input_ids" : self.data[index, :], 
                "lables" : self.labels[index, :]}
    

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    实现torch自带的分布式数据格式:
    1. 在rank 0存储完整数据集
        - 在实际训练过程中, 将各数据调度到各个GPU上
    2. 在rank 0,1,2,3,... 存储各子数据集
        - 则直接取当前rank存在的数据
    3. 以下代码在所有GPU中, 都存储完整数据集, 仅根据一个分布式的sampler实现调度, 但不会出现重叠
    我们以 2 实现一个分布式的数据集
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    N = 256
    dim_in = 16
    dim_hidden = 512
    dim_out = 10

    dataset = MyDataset(N, dim_in, dim_out)
    sampler = DistributedSampler(dataset) # sampler定义了数据分配行为
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    print(dataset.__len__()) # 每个 GPU 存储的数据都一样
    print(dataloader.__len__())

    #  steps * world_size * batch_size = N
    for step, batch in enumerate(dataloader):
        # training ....
        print(rank, step, batch["input_ids"].shape, batch["lables"].shape)
        sleep(0.1)
        
    dist.destroy_process_group()


def run_distributed_dataset(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    实现torch自带的分布式数据格式:
    1. 在rank 0存储完整数据集
        - 在实际训练过程中, 将各数据调度到各个GPU上
    2. 在rank 0,1,2,3,... 存储各子数据集
        - 则直接取当前rank存在的数据
    3. 以下代码在所有GPU中, 都存储完整数据集, 仅根据一个分布式的sampler实现调度, 但不会出现重叠
    我们以 2 实现一个分布式的数据集
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    N = 256
    dim_in = 16
    dim_hidden = 512
    dim_out = 10

    distributed_dataset = MyDataset(N // world_size, dim_in, dim_out)
    dataset = MyDataset(N, dim_in, dim_out)

    if rank == 0:
        distributed_datas = list(dataset.data.split( N // world_size, dim = 0))
        distributed_labels = list(dataset.labels.split( N // world_size, dim = 0))
    else:
        distributed_datas = None
        distributed_labels = None
    dist.scatter(distributed_dataset.data, distributed_datas, src = 0)
    dist.scatter(distributed_dataset.labels, distributed_labels, src = 0)
    dist.barrier()

    dataloader = DataLoader(distributed_dataset, batch_size=2)

    if rank == 0:
    # print(dataset.__len__()) # 每个 GPU 存储的数据都一样
        print(distributed_dataset.__len__()) # 每个 GPU 存储的数据都一样
        print(dataloader.__len__())
        print(dataloader)
    dist.barrier()

    #  steps * world_size * batch_size = N
    for step, batch in enumerate(dataloader):
        # training ....
        print(rank, step, batch["input_ids"].shape, batch["lables"].shape)
        sleep(0.1)
        
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
    mp.spawn(run_distributed_dataset, args=("127.0.0.1", "12801", 4, ), nprocs=4)
