# python pipeline_parallel_basic.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd

class MLP(nn.Module):
    def __init__(self, dim, rank = 0, world_size = 1):
        super(MLP, self).__init__()
        self.dim = dim
        self.rank = rank
        self.world_size = world_size
        self.w1 = nn.Linear(dim, dim * 4, bias = False)
        # self.ReLU = nn.ReLU()
        self.w2 = nn.Linear(dim * 4, dim, bias = False)
    
    def forward(self, x):
        h = self.w1(x)
        # h_act = self.ReLU(h)
        o = self.w2(h)
        return o
    
class PipeModel(nn.Module):
    def __init__(self, dim, num_blocks, rank = 0, world_size = 1):
        super(PipeModel, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.dim = dim
        self.num_blocks = num_blocks
        self.layers = torch.nn.ModuleList()
        self.local_num_blocks = num_blocks // world_size
        for i in range(self.local_num_blocks):
            self.layers.append(MLP(self.dim, self.rank, self.world_size))
        
    def forward(self, x): 
        # x = input.clone()
        for layer in self.layers:
            x = layer(x) + x
        return x


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    参考`custom_gradient_backward.py`可以对中间变量进行反向传播
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    # 准备数据
    dim = 128
    num_blocks = 8
    bs = 8
    if rank == 0:
        x = torch.randn(bs, dim, requires_grad=True)
    else:
        x = torch.zeros(bs, dim, requires_grad=True)
    if rank == world_size-1:
        label = torch.randn(bs, dim).sin()
        loss_fn = nn.MSELoss()

    stage_output = torch.zeros_like(x, requires_grad=True)
    stage_output_grad = torch.zeros_like(x, requires_grad=True)
    

    # 准备模型
    # 标准的实现有两种:
    # 1. 从一个完整的模型 shared, 并根据层数 scatter 对应层数到各GPU上
    # 2. 为了方便我们直接初始化一个 shared 模型
    pipe_model = PipeModel(dim, num_blocks=num_blocks)
    optimizer = optim.SGD(pipe_model.parameters(), lr = 0.0001)


    # forward 
    '''
    前向计算时对于rank 0, 只需要发送到下个gpu rank 1
    而对于rank 1-3 则要先接受上个stage的输出作为当前stage输入, 当前stage计算输出后将结果传给下个Stage
    而rank 3则不需要传给下个Stage
    '''
    for e in range(5):

        optimizer.zero_grad()
        stage_output.grad = None
        stage_output_grad.grad = None
        x.grad = None
        
        if rank == 0:
            print('----pipeline forward---')
        if rank == 0:
            stage_output = pipe_model(x)
            print('forward:', rank)
            dist.send(stage_output, dst = rank+1)
        else:
            dist.recv(x, src = rank-1)
            stage_output = pipe_model(x)
            print('forward:', rank)
            if rank != world_size - 1:
                dist.send(stage_output, dst = rank+1)
        dist.barrier()
        
        # backward
        '''
        梯度更新, 除了最后一个rank计算MSE, 则传递的梯度为x.grad, 接受的梯度stage_output_grad
        '''
        if rank == 0:
            print('----pipeline backward---')
        if rank == world_size - 1:
            # print(stage_output)
            loss = loss_fn(stage_output, label)
            print(loss)
            loss.backward() # 考虑如何避免backward过程中, 各个 stage 之间会有依赖
            print('backward:', rank)
            dist.send(x.grad.clone(), dst = rank - 1)
        else:
            dist.recv(stage_output_grad, src = rank + 1)
            stage_output.backward(gradient = stage_output_grad)
            print('backward:', rank)
            if rank != 0:
                dist.send(x.grad.clone(), dst = rank - 1)
            
        dist.barrier()
        
        # 统一更新梯度
        # if rank == 3:
        #     print(pipe_model.layers[0].w1.weight.grad[0,:4])
        #     print(pipe_model.layers[0].w1.weight[0,:4])
        optimizer.step()
        # dist.barrier()
        # if rank == 3:
            # print(pipe_model.layers[0].w1.weight.grad[0,:4])
            # print(pipe_model.layers[0].w1.weight[0,:4])
        # optimizer.zero_grad()
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
