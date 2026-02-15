# torchrun  adam_mix_precision.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    '''
    (1) 参数权重和梯度都需要 16bit 进行存储
    (2) 参数权重需要有 32bit 参数进行备份, 用于更新参数
    '''

    def __init__(self):
        super(ToyModel, self).__init__()
        self.w1 = nn.Linear(128, 512, bias=False, dtype=torch.float32)
        self.w2 = nn.Linear(512, 10, bias=False, dtype=torch.float32)  # 用于备份
        self.w1_bf16 = nn.Linear(128, 512, bias=False, dtype=torch.bfloat16)
        self.w2_bf16 = nn.Linear(512, 10, bias=False, dtype=torch.bfloat16)

        self.w1_bf16.weight.data = self.w1_bf16.weight.data.clone().to(torch.bfloat16)
        self.w2_bf16.weight.data = self.w2_bf16.weight.data.clone().to(torch.bfloat16)


    def forward(self, x):
        '''
        bf16矩阵乘法, 可以看成是 y = sum_i (wi xi), 
        精度累加优化实现: y = fp16[sum_i (fp32(wi) fp32(xi))], 
        '''
        hidden = self.w1_bf16(x)
        output = self.w2_bf16(hidden)
        return output, hidden


def train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        outputs, _ = model(input)
        optimizer.zero_grad()

        loss = loss_fn(outputs, labels)

        # 混合精度时需要将损失进行放大, 再更新梯度
        # 优化: 动态缩放因子
        scale = 10.0
        loss_scale = loss * torch.tensor([scale], dtype=torch.bfloat16)

        loss_scale.backward()
        optimizer.step(scale=scale) 

        if rank == 0:
            if i % 10 == 0:
                print(loss)
            # print(model.w1.weight.data[0,:4])


class MixPrecisionAdam:
    def __init__(self, params, scale=10, lr=1e-3, beta1=0.90, beta2=0.999,  eps=1e-8):
        super(MixPrecisionAdam, self).__init__()
        self.eps = eps
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0.0
        
        dict(params)

        # 将优化器参数限定为 bf16 参数, fp32 参数不参与计算, 只用于备份和更新
        self.params = []
        self.params_target = []
        for name, param in params:
            if param.requires_grad:
                self.params.append(param)  # 只保留 bf16 参数参与计算
                name_target = name.replace('_bf16', '') # w1_bf16 -> w1
                print(name, name_target)
                self.params_target.append(dict(params)[name_target])

        self.M = [torch.zeros_like(param.data, dtype=torch.float32)
                  for param in self.params]
        self.V = [torch.zeros_like(param.data, dtype=torch.float32)
                  for param in self.params]

    def step(self, weight_decay=1e-2, scale=10.0):
        self.t += 1
        # with torch.no_grad():
        for param, param_target, M, V in zip(self.params, self.params_target, self.M, self.V):
            if param.grad == None:
                continue  # fp 32 不参与 forward 计算就不会有梯度, 直接跳过

            param.grad *= scale

            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)

            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            # 1. optimizer 内部更新 fp32 参数,
            param_target -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
            
            # 2. optimizer 更新 bf16 参数
            param.data = param_target.data.clone().to(torch.bfloat16)  # 同步更新 bf16 参数

    def zero_grad(self, ):
        for param in self.params:
            if param.grad != None:
                param.grad *= torch.zeros_like(param.grad, dtype=torch.bfloat16)


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    混合精度训练流程
    1. 初始化fp32参数备份, 并创建 bf16 参数用于计算
    2. 梯度计算为 bf16, 优化器参数为 fp32, 特别的我们可以将优化器参数计算放置在CPU上来计算(减少显存)
    3. 为了减少 bf16 训练的下溢影响, 将损失乘于scale, 对梯度除于scale, 保留与参数同等量纲
    4. 更新 fp32 参数, 并同步给 bf16 参数
    '''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank,
                            world_size=world_size)

    model = ToyModel()
    loss_fn = nn.MSELoss()

    # 限定只有 bf16 参数参与计算, fp32 参数不参与计算, 只用于备份和更新
    for name, param in model.named_parameters():
        if 'bf16' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = MixPrecisionAdam(model.named_parameters(), lr=0.0001, scale=10)

    N = 128
    input = torch.randn(N, 128, dtype=torch.bfloat16)
    labels = torch.randn(N, 10, dtype=torch.bfloat16)

    epochs = 1000
    if rank == 0:
        print('-'*100)
        print('USE pytorch ddp training')
    train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 1, ), nprocs=1)
