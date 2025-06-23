# python adam_zero3.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP

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
    
    def forward_zero3(self, x):
        '''
        前向计算会遇到问题:
        1. 是否是即时gather w, 算完后, 再将冗余的 w1 丢弃？
        2. 逐层 gather w, 一次性前向算完后, 并且完成backward后, 再移除冗余的weight ✅
        这里需要独立存储原tensor维度, 用于恢复完整的weight, 再进行前向计算
        '''
        w1_gather = torch.zeros(self.shape_list[0])
        dist.all_gather_into_tensor( w1_gather.view(-1), self.w1.weight.data, )
        self.w1.weight.data = w1_gather
        hidden = self.w1(x)

        w2_gather = torch.zeros(self.shape_list[1])
        dist.all_gather_into_tensor( w2_gather.view(-1), self.w2.weight.data, )
        self.w2.weight.data = w2_gather
        output = self.w2(hidden)

        return output, hidden


    
def backward_zero3(model, loss, rank, world_size):
    '''
    zero 3 实现: 前向过程中我们仍然看成前向是完整参数的forward, 反算如下:
    2. 对应的backward gradient 也就是完整的
    3. 遵照ZeRO-2 完成梯度切分
    4. 将冗余参数移除
    '''
    # step 1:
    # loss.backward() # reduce 

    # step 2:
    for param in model.parameters():
        if param.grad != None:
            
            # all-reduce
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            dist.barrier()
            # param.grad = param.grad / world_size
            tmp_param = deepcopy(param.grad) / world_size

            # scatter
            shared_size = param.grad.numel() // world_size #假设都能整除
            param.grad.data = torch.zeros(shared_size)

            if rank == 0:
                grad_list = list(torch.split(tmp_param.view(-1), shared_size))
            else:
                grad_list = []
            
            dist.scatter(param.grad.data, grad_list, src = 0)
    # # step 3: 当梯度计算完
    # model.shared_model()


def train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        optimizer.zero_grad()
        # outputs, _ = model(input)
        outputs, _ = model.forward_zero3(input) # 使用分布式版本的前向计算

        loss = loss_fn(outputs, labels)
        loss.backward()  # 替换为以下方法来更新梯度
        dist.barrier()
        model.shared_model()
        backward_zero3(model, loss, rank, world_size)   
        optimizer.step()  
        dist.barrier()
        if rank == 0:
            if i % 10 == 0:
                print(loss)


class MyAdamZeRO3:
    '''
    由于模型的参数已经 shared 过, 那么就直接在 shared weight 上建立 adam 参数
    '''
    def __init__(self, 
                 params, 
                 lr = 1e-3, 
                 beta1 = 0.90, 
                 beta2 = 0.999,  
                 eps=1e-8, 
                 world_size=1, 
                 rank = 0):
        super(MyAdamZeRO3, self).__init__()
        self.eps = eps
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = list(params)
        self.t = 0.0
        self.world_size = world_size
        self.rank = rank 

        self.M = []
        self.V = []

        # 对每层数据初始化固定的优化器参数, 由于每个rank的参数量相等，
        # 直接零初始化, 而不需要切分再发送到其他GPU上
        for param in self.params:
            # shared_size = param.data.numel() // self.world_size #假设都能整除
            self.M.append( torch.zeros(param.data.shape, dtype = torch.float32) )
            self.V.append( torch.zeros(param.data.shape, dtype = torch.float32) )

        
    def step(self, weight_decay=1e-2):
        '''
        1. 对块参数进行更新
        2. 对各块 all-gather 一致化参数
        '''
        self.t += 1
        # with torch.no_grad():
        for param, M, V in zip(self.params, self.M, self.V):

            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)

            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

            # ZeRO-3 不需要同步参数

    def zero_grad(self, ):
        for param in self.params:
            if param.grad != None:
                # param.grad = torch.zeros_like(param.grad) 
                param.grad = None

    

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    ZeRO-3 关键实现
    1. 将模型参数进行切分
        - 将完整模型加载到GPU, 再进行切分
        - 加载模型时就开始 shared 处理, 避免加载的存储峰值
    2. 模型在前向计算时, 需要all-gather 层参数, 才能完成前向计算, 这是与 tensor parallellism 的最大区别
    3. 反向过程与 ZeRO-2 唯一区别在于 不需要 all-gather 参数了.
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = ToyModel(world_size = world_size, rank = rank) # 初始化一个分布式模型
    model.shared_model()
    print(model.w1.weight.shape)
    print(model.w2.weight.shape)
    loss_fn = nn.MSELoss()
    optimizer = MyAdamZeRO3(model.parameters(), lr=0.001, world_size = world_size, rank = rank)

    N = 128
    input = torch.randn(N, 128)
    labels = torch.randn(N, 10)

    epochs = 1000
    if rank == 0:
        print('-'*100)
        print('ZeRO3 training')
    train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs)

    # 每张卡上的参数是不同的
    print(rank , model.w1.weight.data)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
