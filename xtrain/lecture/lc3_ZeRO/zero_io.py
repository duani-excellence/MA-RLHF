# python zero_io.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class ToyModel(nn.Module):
    def __init__(self, world_size, rank):
        super(ToyModel, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.w1 = nn.Linear(128, 512, bias = False)
        self.w2 = nn.Linear(512, 10, bias = False)
        self.shape_list = [ self.w1.weight.data.shape, 
                            self.w2.weight.data.shape, ]

    def forward(self, x):
        hidden = self.w1(x)
        output = self.w2(hidden)
        return output, hidden
    
    def shared_model(self,):
        for param in self.parameters():
            shared_size = param.data.numel() // self.world_size #假设都能整除
            weight_data = deepcopy(param.data).view(-1)
            param.data = torch.zeros(shared_size)
            if self.rank == 0:
                scatter_list = list(weight_data.split(shared_size, dim = 0))
            else:
                scatter_list = None
            dist.scatter(param.data, scatter_list, src = 0)

    def gather_model(self, ):
        w1_gather = torch.zeros(self.shape_list[0])
        dist.all_gather_into_tensor( w1_gather.view(-1), self.w1.weight.data, )
        self.w1.weight.data = w1_gather
        w2_gather = torch.zeros(self.shape_list[1])
        dist.all_gather_into_tensor( w2_gather.view(-1), self.w2.weight.data, )
        self.w2.weight.data = w2_gather
    
    

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    在使用ZeRO-3时, 由于参数是分块的, 对于IO分为:
        A.存储: (1) 每个rank存储独立的参数 (2) 将参数归并到一个rank上, 存储完整参数
        B.读取: (1) 每个rank加载独立的参数 (2) 将参数读取到一个rank上, 再进行分散
        根据复杂度, 我们分别实现 A(1) 和 B(1)
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = ToyModel(world_size = world_size, rank = rank) # 初始化一个分布式模型

    X = torch.ones(1, 128) # 输入

    file_name = "./../../output/model.bin"
    current_dir = os.getcwd()
    file_name_abs = os.path.join(current_dir, file_name)

    # 单卡保存模型
    if rank == 0:
        result, _  = model(X)
        print('original model output', result[0,:4])
        torch.save(model.state_dict(), file_name_abs)
    dist.barrier()
    
    # 所有卡都加载同个模型
    model.load_state_dict(torch.load(file_name_abs))
    dist.barrier()
    print(model)

    # model.load_state_dict()

    # A(1). 保存模型切片
    #  - 切分模型
    #  - 各卡存储相同模型
    file_name = "./../../output/shared_model/" + str(rank) + '.bin'
    file_name_abs = os.path.join(current_dir, file_name)

    model.shared_model() # 切分模型
    print(model.w1.weight.data[:4]) 
    torch.save(model.state_dict(), file_name_abs) # 保存模型
    dist.barrier()

    # B(1). 分块加载模型
    model.load_state_dict(torch.load(file_name_abs))
    dist.barrier()

    # 将模型合并
    model.gather_model()

    if rank == 0:
        result, _  = model(X)
        print('shared model load output', result[0,:4])
    
    dist.destroy_process_group()
    



def dynamic_shared_load(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    当我们从 huggingface 下载的模型通常有两种
        - (1) 完整参数, 那么我们可以分层载入参数, 切片并发送到各个GPU上
        - (2) 云端的模型本身就是分块的参数, 分块是 layer-wise 的, 所以仍然复用(1)
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = ToyModel(world_size = world_size, rank = rank) # 初始化一个分布式模型

    X = torch.ones(1, 128) # 输入

    file_name = "./../../output/model.bin"
    current_dir = os.getcwd()
    file_name_abs = os.path.join(current_dir, file_name)

    # 单卡保存模型
    if rank == 0:
        result, _  = model(X)
        print('original model output', result[0,:4])
        torch.save(model.state_dict(), file_name_abs)
    dist.barrier()
    
    
    dist.destroy_process_group()
    


if __name__ == '__main__':

    # 创建 output 目录
    current_dir = os.getcwd()
    dir_name = "./../../output/"
    abs_output_dir = os.path.join(current_dir, dir_name)
    abs_output_dir = os.path.dirname(abs_output_dir)
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)

    # 创建 output/shared_model 目录
    current_dir = os.getcwd()
    dir_name = "./../../output/shared_model"
    abs_output_dir = os.path.join(current_dir, dir_name)
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)

    # 实现 完整参数/切块参数 的保存和加载
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)

    # 实现 完整参数保存, 分块参数逐层动态加载, 避免一次性载入导致爆显存
    mp.spawn(dynamic_shared_load, args=("127.0.0.1", "12801", 4, ), nprocs=4)
