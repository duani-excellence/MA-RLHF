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


class MyAdamZeRO1:
    '''
    初始化对优化器参数flatten化, 并分块到各rank上
    这种做法的好处在于 Adam 优化器参数更新是 element-wise 的
    '''
    def __init__(self, 
                 params, 
                 lr = 1e-3, 
                 beta1 = 0.90, 
                 beta2 = 0.999,  
                 eps=1e-8, 
                 world_size=1, 
                 rank = 0):
        super(MyAdamZeRO1, self).__init__()
        self.eps = eps
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = list(params)
        self.t = 0.0
        self.world_size = world_size
        self.rank = rank 

        self.M = [ ]
        self.V = [ ]

        # 对每层数据初始化固定的优化器参数, 由于每个rank的参数量相等，
        # 直接零初始化, 而不需要切分再发送到其他GPU上
        for param in self.params:
            shared_size = param.data.numel() // self.world_size #假设都能整除
            self.M.append( torch.zeros(shared_size, dtype = torch.float32) )
            self.V.append( torch.zeros(shared_size, dtype = torch.float32) )

        
    def step(self, weight_decay=1e-2):
        '''
        1. 对梯度 reduce
        2. 对块参数进行更新
        3. 对各块 all-gather 一致化参数
        '''
        self.t += 1
        # with torch.no_grad():
        for param, M, V in zip(self.params, self.M, self.V):

            # 将梯度进行 all reduce
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= self.world_size

            shared_size_grad = param.grad.numel() // self.world_size #假设都能整除
            shared_grad = param.grad.view(-1)[self.rank * shared_size_grad : (self.rank + 1) * shared_size_grad]

            M = self.beta1 * M + (1 - self.beta1) * shared_grad
            V = self.beta2 * V + (1 - self.beta2) * shared_grad.pow(2)

            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            weight_data = param.data.view(-1)[self.rank * shared_size_grad : (self.rank + 1) * shared_size_grad] 
            weight_data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

            # 同步参数
            # all-gather
            gather_tensor = torch.zeros( param.grad.numel(), dtype = param.data.dtype)
            dist.all_gather_into_tensor(gather_tensor, weight_data)
            param.data = gather_tensor.reshape(param.data.shape)
            dist.barrier()


    def zero_grad(self, ):
        for param in self.params:
            if param.grad != None:
                param.grad *= torch.zeros_like(param.grad)
                

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    分布式 adam 优化器对象, 实现有多种存储方式:
        - 每个rank有 adam 优化器: 存储量xN倍
        - rank 0 有 adam 优化器: 那么每张卡rank 存储/计算 不均衡, 但完整系统仅存储一份优化器参数
        - [ZeRO] 将 adam 优化器参数均分到各 gpu 上, 那么存储/计算 均衡？是否还有其他的资源消耗, 如通信
    我们实现 ZeRO Adam 优化器, 更准确的是实现 ZeRO-1 优化.
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = ToyModel() # 假设在多卡环境下, 每个机器初始化的模型参数是一致的
    loss_fn = nn.MSELoss()
    optimizer = MyAdamZeRO1(model.parameters(), lr=0.001, world_size = world_size, rank = rank)

    N = 128
    input = torch.randn(N, 128)
    labels = torch.randn(N, 10)

    epochs = 1000
    if rank == 0:
        print('-'*100)
        print('ZeRO1 training')
    train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs)

    # 检验参数是否是同步的
    print(rank , model.w1.weight.data)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
