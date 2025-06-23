# python zero_io_shared_load.py

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
import os


class SharedToyModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, is_shared, world_size, rank):
        super(SharedToyModel, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.is_shared = is_shared
        if self.is_shared:
            self.w1 = nn.Linear(dim_in * dim_hidden // world_size, 1, bias = False)
            self.w1.weight.data = self.w1.weight.data[0]
            self.w2 = nn.Linear(dim_hidden * dim_out // world_size, 1, bias = False)
            self.w2.weight.data = self.w2.weight.data[0]
        else:
            self.w1 = nn.Linear(dim_in , dim_hidden , bias = False)
            self.w2 = nn.Linear(dim_hidden , dim_out , bias = False)
        self.shape_list = [ torch.Size([dim_in, dim_hidden]), 
                            torch.Size([dim_hidden, dim_out]), ]
    def forward(self, X):
        if self.is_shared:
            '''
            那么 forward 过程需要gather, 分两种情况:
            1. 逐层gather, 计算, 释放, 这里会存在释放会导致无法正常backward, 逐层gather也会产生频繁通信
                另外可能需要魔改 autograd, 或者手动实现 backward
            2. 一次性gather, 再计算, backward, 释放, 这样显存峰值 > 完整weight + 完整 grad
            '''
            self.gather_model()
        result = self.w2(self.w1(X))
        return result
        
    def gather_model(self, ):
        # self.is_shared = False
        w1_gather = torch.zeros(self.shape_list[0])
        dist.all_gather_into_tensor( w1_gather.view(-1), self.w1.weight.data, )
        self.w1.weight.data = w1_gather
        w2_gather = torch.zeros(self.shape_list[1])
        dist.all_gather_into_tensor( w2_gather.view(-1), self.w2.weight.data, )
        self.w2.weight.data = w2_gather

            
    
    


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    当我们从 huggingface 下载的模型通常有两种
        - (1) 完整参数, 那么我们可以分层载入参数, 切片并发送到各个GPU上
        - (2) 云端的模型本身就是分块的参数, 分块是 layer-wise 的, 所以仍然复用(1)
    我们期望的实现方式是
        1. 在CPU上初始化出一个完整模型1, 但是会分配完整大小的存储空间, 所以加载一个字典`state_dict`替代模型
        2. 在各GPU上初始化出分块模型
        3. 将字典逐层切分写到 GPU 里
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    dim_in = 128
    dim_hidden = 512
    dim_out = 10
    model = SharedToyModel(dim_in = dim_in,
                     dim_hidden = dim_hidden,
                     dim_out = dim_out,
                     is_shared = False,
                     world_size = world_size, 
                     rank = rank) # 初始化一个分布式模型

    X = torch.ones(1, dim_in) # 输入

    file_name = "./../../output/model.bin"
    current_dir = os.getcwd()
    file_name_abs = os.path.join(current_dir, file_name)

    # 单卡保存模型
    if rank == 0:
        # result, _  = model(X)
        # print('original model output', result[0,:4])
        torch.save(model.state_dict(), file_name_abs)
    dist.barrier()

    del model

    # 在各个GPU上初始化出一个分块的模型, 将 `is_shared = True`
    model = SharedToyModel(dim_in = dim_in,
                    dim_hidden = dim_hidden,
                    dim_out = dim_out,
                    is_shared = True,
                    world_size = world_size, 
                    rank = rank) # 初始化一个分布式模型
    print(model.w1.weight.shape)
    # if rank == 0:
    state_dict = torch.load(file_name_abs)

    for name, param in model.state_dict().items():
        print(name)
        if rank == 0:
            if name in state_dict:
                data = deepcopy(state_dict[name].view(-1))
                shared_size = data.shape[0] // world_size
                data_list = list(torch.split(data, shared_size))
        else:
            data_list = []
        if name in state_dict:
            dist.scatter(param.data, data_list, src = 0)
        
    print(model.w1.weight)
        
    dist.destroy_process_group()
    


if __name__ == '__main__':

    # 创建 output 目录
    current_dir = os.getcwd()
    dir_name = "./../../output/"
    abs_output_dir = os.path.join(current_dir, dir_name)
    abs_output_dir = os.path.dirname(abs_output_dir)
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)

    # 实现 完整参数保存, 分块参数逐层动态加载, 避免一次性载入导致爆显存
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
