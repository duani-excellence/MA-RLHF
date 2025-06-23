# ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#
# run with
# torchrun  torch_ddp.py
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

def train_ddp(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model(input)
        loss = loss_fn(outputs, labels)
        loss.backward()          
        if rank < 2:
            if i % 1 == 0:
                # 表明不同 rank 处理不同数据, 得到不一样的 loss
                print(f'rank {rank}:', 'loss:', loss.item())
                # 查看 graient 值
                print(f'rank {rank}:', 'grad_w1:', model.module.w2.weight.grad[0,:4])
        optimizer.step()  # 非分布式版本


def train_my_ddp(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    '''
    手动实现多层 gradient 传播, 思考两种方案
        1. 是否每个rank独立backward, 再统一聚合梯度
        2. 是否逐层 backward, all-reduce, backward, all-reduce... 直至结束?
            对dw, dx是否同样操作?
        实际实现是1, 每个 rank 上, 独立反传所有层的梯度, 而在最终对每层梯度做聚合.
        另外也可以实现一个分布式版本的optimizer来完成梯度reduce和更新
    '''
    lr = optimizer.param_groups[0]['lr']
    bs, _ = input.shape
    _, dim_out = labels.shape
    with torch.no_grad():
        for i in range(epochs):
            # optimizer.zero_grad()
            output, hidden = model(input)
            # torch 的 MSE 实际为 (output - label) ** 2, 不会再增加 1/2 系数
            loss = loss_fn(output, labels) 

            do = 2 * (output - labels) / ( bs * dim_out ) # bs x dim_out
            grad_w2 = do.t() @ hidden
            grad_hidden = do @ model.w2.weight            # hidden不需要聚合
            dist.all_reduce(grad_w2, dist.ReduceOp.SUM)
            grad_w2 = grad_w2 / world_size
            model.w2.weight -= lr *  grad_w2

            grad_w1 = grad_hidden.t() @ input
            dist.all_reduce(grad_w1, dist.ReduceOp.SUM)
            grad_w1 = grad_w1 / world_size
            model.w1.weight -= lr *  grad_w1

            if rank < 2:
                if i % 1 == 0:
                    # 表明不同 rank 处理不同数据, 得到不一样的 loss
                    print(f'rank {rank}:', 'loss:', loss.item())
                    # 查看 graient 值
                    print(f'rank {rank}:', 'grad_w1:', grad_w2[0,:4])
                
        optimizer.step()  # 非分布式版本



def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    torch 自带数据并行模块, 各个 rank 上
        1. 参数同步
        2. 数据不同
        3. loss不同
        4. graident相同
        5. optimizer后更新的参数相同
    因此, 官方的ddp 需要实现 gradient 的 all-reduce
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    
    model = ToyModel()
    ddp_model = DDP(model) # 官方DDP
    my_model = deepcopy(model) # 手写DDP
    print(f'rank {rank}:', ddp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    optimizer.zero_grad()

    N = 8
    input = torch.randn(N, 128)
    labels = torch.randn(N, 10)
    print(f'rank {rank}:', input[rank,:4]) # 每个rank的数据是不同的

    # 使用官方的实现
    epochs = 10
    if rank == 0:
        print('-'*100)
        print('USE pytorch ddp training')
    train_ddp(rank, world_size, ddp_model, input, labels, loss_fn, optimizer, epochs)
    
    if rank == 0:
        print('-'*100)
        print('USE pytorch MY ddp training')
    train_my_ddp(rank, world_size, my_model, input, labels, loss_fn, optimizer, epochs)

    if rank == 0:
        print('-'*100)
        print('result')
    print(f'rank {rank}: my model', my_model.w1.weight.data[0,:4])
    print(f'rank {rank}: ddp model', ddp_model.module.w1.weight.data[0,:4])

    # 期望每个 rank 上的模型参数一致

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
