# python adam.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP

class MyGradFuntion(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        # output = w @ input.t() # 2x4 = 2x3 @ 3x4
        output = input @ w # 2x4 = 2x3 @ 3x4
        print(output.shape)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        代码仅演示 backward 过程里, 运用分布式操作来聚合梯度
        '''
        w, input = ctx.saved_tensors
        # print(grad_output)
        grad_input = grad_output @ w.t()
        grad_w = input.t() @ grad_output
        print('before gather:', grad_w[0,:4])
        dist.all_reduce(grad_w, dist.ReduceOp.SUM)
        print('after gather:', grad_w[0,:4])
        return  grad_input.t(), grad_w 
    
class MyGradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(64, 8, bias = False)
        dist.broadcast(self.w.weight.data, src = 0)

    def forward(self, x):
        return MyGradFuntion.apply(x, self.w.weight.t())   
    

def train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        outputs  = model(input) # 结果要转置
        optimizer.zero_grad()

        loss = loss_fn(outputs, labels)
        loss.backward()     
        optimizer.step()  
        if rank == 0:
            if i % 1 == 0:
                print(loss)

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    在张量并行过程中, 参照`tensor_parallel_easy.py`实现的 列并行 梯度计算时,
    对于求输入 x 的梯度时, 需要 recude 各个GPU上的 dx 梯度, 才能将梯度进一步传播,
    这与数据并行(dp)有本质区别, dp可以一次性backward完, 再算梯度的 reduce
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = MyGradModel() # 假设在多卡环境下, 每个机器初始化的模型参数是一致的
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(rank , model.w.weight.data[0,:4])

    N = 128
    input = torch.randn(N, 64)
    labels = torch.randn(N, 8)


    epochs = 1
    if rank == 0:
        print('-'*100)
        print('distributed custom gradient')
    train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs)

    # 检验参数是否是同步的
    print(rank , model.w.weight.data[0,:4])
    
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
