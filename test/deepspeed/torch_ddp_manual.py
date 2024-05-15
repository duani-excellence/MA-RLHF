#!/usr/bin/env python
# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_ddp_manual.py

# 在一个原生model上做分布式并行训练， 此时不使用DistributedDataParallel类
# 手工创建数据集， 数据集是否创建了两个？

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
import time


from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        # 在此处对样本和标签进行预处理或其他操作
        # 返回预处理后的样本和标签
        return sample, label


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])



class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def partition_dataset():
    N = 128
    input = torch.randn(N, 10)
    labels = torch.randn(N, 5)
    print(input)

    dataset = CustomDataset(input, labels)

    # print(dataset)

    size = dist.get_world_size()
    bsz = N // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if type(param) is torch.Tensor:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= size


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = ToyModel()
    model = model
#    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.MSELoss()

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(100):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = loss_fn(output, target)
            epoch_loss += loss

            loss.backward()

            print('-'*50)
            time.sleep(0.5)
            print(f'rank{dist.get_rank()}, loss:{loss}') # 不同卡得到不同的loss, 那么就有不同的梯度

            # 手工 reduce 梯度
            average_gradients(model) # 这里的梯度就是手工来求平均

            optimizer.step()


        time.sleep(0.5)
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

        # time.sleep(0.5)

    dist.destroy_process_group()

def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    # 手工创建进程？
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
