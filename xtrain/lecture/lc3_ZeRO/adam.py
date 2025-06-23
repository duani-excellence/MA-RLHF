# torchrun  adam.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.w1 = nn.Linear(128, 512, bias = False)
        self.w2 = nn.Linear(512, 10, bias = False)

    def forward(self, x):
        hidden = self.w1(x)
        output = self.w2(hidden)
        return output, hidden

def train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        outputs, _ = model(input)
        optimizer.zero_grad()

        loss = loss_fn(outputs, labels)
        loss.backward()     
        optimizer.step()  
        if rank == 0:
            if i % 10 == 0:
                print(loss)
            # print(model.w1.weight.data[0,:4])


class MyAdam:
    def __init__(self, params, lr = 1e-3, beta1 = 0.90, beta2 = 0.999,  eps=1e-8):
        super(MyAdam, self).__init__()
        self.eps = eps
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = list(params) # 引用, 不占用存储, 优化器内部更新参数, 外部的模型参数也会同时改变
        self.t = 0.0
        self.M = [ torch.zeros_like(param.data, dtype=torch.float32) for param in self.params ]
        self.V = [ torch.zeros_like(param.data, dtype=torch.float32) for param in self.params ]

    def step(self, weight_decay=1e-2):
        self.t += 1
        # with torch.no_grad():
        for param, M, V in zip(self.params, self.M, self.V):
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)

            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            # if weight_decay is not None: # adamW
            #     param =  param.weight - self.lr * (m_hat / (v_hat.sqrt() + self.eps) + weight_decay * param.weight) adamw
            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
    
    def zero_grad(self, ):
        for param in self.params:
            if param.grad != None:
                param.grad *= torch.zeros_like(param.grad)
                

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    1. 手写实现 adam 优化器
    2. 分布式adam 优化器面临问题, 
        - 是将 gradient先 reduce 再算参数更新量？  ✅
        - 先计算各rank的更新量, 再将 更新量 reduce?
    
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = ToyModel()
    loss_fn = nn.MSELoss()
    optimizer = MyAdam(model.parameters(), lr=0.0001)

    N = 128
    input = torch.randn(N, 128)
    labels = torch.randn(N, 10)

    epochs = 1000
    if rank == 0:
        print('-'*100)
        print('ADAM')
    train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 1, ), nprocs=1)
