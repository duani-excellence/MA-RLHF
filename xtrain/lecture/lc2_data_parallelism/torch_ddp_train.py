# ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#
# run with
# torchrun  torch_ddp_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, classes):
        super(ToyModel, self).__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden, bias = True)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(dim_hidden, classes, bias = True)

    def forward(self, x):
        hidden = self.relu(self.w1(x))
        output = self.w2(hidden)
        return output, hidden


class MyDataset(Dataset):

    def __init__(self, N, dim, classes):
        super().__init__()
        self.data = torch.randn(N, dim, dtype=torch.float32)
        self.labels = torch.randn(N, classes, dtype=torch.float32) 
        self.labels = torch.nn.functional.softmax(self.labels, dim = -1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {"input_ids" : self.data[index, :], 
                "lables" : self.labels[index, :]}
    

def train_ddp(rank, world_size, model, dataloader, loss_fn, optimizer, epochs):
    for i in range(epochs):
        for k, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs, _ = model(batch["input_ids"])
            loss = loss_fn(outputs, batch["lables"])
            loss.backward()
            optimizer.step()  
            dist.all_reduce(loss.data, dist.ReduceOp.SUM)
        if rank == 0:
            # if k % 10 == 0:
            print(i, loss.data / world_size)


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    实现一个分布式训练程序
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    N = 1024
    dim_in = 16
    dim_hidden = 512
    dim_out = 10

    model = ToyModel(dim_in, dim_hidden, dim_out)
    ddp_model = DDP(model) # 官方DDP
    print(f'rank {rank}:', ddp_model)

    dataset = MyDataset(N, dim_in, dim_out)
    sampler = DistributedSampler(dataset) # sampler定义了数据分配行为
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 使用官方的实现
    epochs = 100
    if rank == 0:
        print('-'*100)
        print('USE pytorch ddp training')

    train_ddp(rank, world_size, ddp_model, dataloader, loss_fn, optimizer, epochs)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
