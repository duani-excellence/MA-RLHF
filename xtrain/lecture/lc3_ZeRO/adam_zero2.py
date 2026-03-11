# python adam_zero2.py

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
    
def backward_zero2(model, loss, rank, world_size):
    '''
    zero 2 关键在于梯度切分, 在反向时涉及1. reduce 梯度, 2. 将梯度在分散到各GPU中
    实现有两种方法:
    1. 边反向传播, 边 Scatter 梯度
    2. 一次性完成 反向传播, 再逐层 Scatter 梯度
    为了实现方便采用方法2, 方法 1 涉及到 backward() 内部函数的管理
    '''
    # loss.backward() # reduce 


    for param in model.parameters():
        if param.grad != None:
            
            # all-reduce
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= world_size
            tmp_param = deepcopy(param.grad)

            # scatter
            shared_size = param.grad.numel() // world_size #假设都能整除
            param.grad.data = torch.zeros(shared_size)

            if rank == 0:
                grad_list = list(torch.split(tmp_param.view(-1), shared_size))
                dist.scatter(param.grad.data, grad_list, src = 0)
            else:
                dist.scatter(param.grad.data, [], src = 0)


def train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        outputs, _ = model(input)
        optimizer.zero_grad()

        loss = loss_fn(outputs, labels)
        loss.backward()  # 替换为以下方法来更新梯度
        backward_zero2(model, loss, rank, world_size)   
        optimizer.step()  
        if rank == 0:
            if i % 10 == 0:
                print(loss)


class MyAdamZeRO2:
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
        super(MyAdamZeRO2, self).__init__()
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
        1. 对块参数进行更新
        2. 对各块 all-gather 一致化参数
        '''
        self.t += 1
        # with torch.no_grad():
        for param, M, V in zip(self.params, self.M, self.V):

            shared_size = param.data.numel() // self.world_size #假设都能整除
            # shared_grad = param.grad.view(-1)[self.rank * shared_size_grad : (self.rank + 1) * shared_size_grad]

            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)

            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            weight_data = param.data.view(-1)[self.rank * shared_size : (self.rank + 1) * shared_size] 
            weight_data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

            # 同步参数
            # all-gather
            gather_tensor = torch.zeros( param.data.numel(), dtype = param.data.dtype)
            dist.all_gather_into_tensor(gather_tensor, weight_data)
            param.data = gather_tensor.reshape(param.data.shape)
            dist.barrier()

    def zero_grad(self, ):
        for param in self.params:
            if param.grad != None:
                param.grad = torch.zeros_like(param.data) 
                

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    ZeRO-2 在 1 上增加对 梯度的切分
    本代码手动实现 梯度的 reduce-scatter
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = ToyModel()
    loss_fn = nn.MSELoss()
    optimizer = MyAdamZeRO2(model.parameters(), lr=0.001, world_size = world_size, rank = rank)

    N = 128
    input = torch.randn(N, 128)
    labels = torch.randn(N, 10)

    epochs = 1000
    if rank == 0:
        print('-'*100)
        print('ZeRO2 training')
    train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs)

    # 检验参数是否是同步的
    print(rank , model.w1.weight.data)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
