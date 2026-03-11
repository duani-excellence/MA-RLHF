# python pipeline_parallel_gpipe.py
# gpipe主要实现:
# 1. micro-batch: 基础的PP bubble率高, 使用micro-batch版本的PP则可以减少bubble率
# 2. 梯度累积: 由于micro-batch存在, 可以多次backward累积(torch自动), 一次做参数更新
# 3. gradient-checkpoint: 独立实现

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from pipeline_parallel_basic import PipeModel

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
    bs = 16
    mini_batch = world_size
    mini_batch_size = bs // world_size
    if rank == 0:
        x = torch.randn(bs, dim, requires_grad=True)
    else:
        x = torch.zeros(bs, dim, requires_grad=True)
    x_list = list(torch.chunk(x, world_size, dim = 0))


    if rank == world_size-1:
        label = torch.randn(bs, dim).sin()
        label_list =  list(torch.chunk(label, world_size, dim = 0))
        loss_fn = nn.MSELoss()

    stage_output = torch.zeros_like(x_list[0], requires_grad=True)
    stage_output_grad = torch.zeros_like(x, requires_grad=True)
    stage_output_list = list(torch.chunk(stage_output, world_size, dim = 0))
    stage_output_grad_list = list(torch.chunk(stage_output_grad, world_size, dim = 0))
    # stage_output_list = []
    # stage_output_grad_list = []
    

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
    实现上我们考虑异步发送, 而接受是同步, 仅考虑4个时间步:
    '''
    for e in range(1):

        optimizer.zero_grad()
        stage_output.grad = None
        stage_output_grad.grad = None
        x.grad = None
        
        if rank == 0:
            print('----pipeline forward---')

        '''
        rank1,rank2,rank3 由于没有收到前rank的结果则阻塞 | rank2,3,4 t <- 0
        [异步发送 isend] rank0, 计算出y0a 发送给 rank1. |  rank0 t <- 1

        rank2, rank3 由于没有收到前rank的结果则阻塞 | rank 3,4 t <- 0
        rank1 接收到 rank0 的 y0a, 计算出 y1a 发送给 rank2  | rank2 t <- 1
        [异步发送 isend] rank0 计算出y0b 发送给 rank1. | rank0 t <- 2
        
        这个过程可以发现每个GPU的执行步骤(t)并不同, 如果接受不到数据则在循环里会阻塞(recv)
        '''
        reqs = []
        for i in range(world_size):
            if rank != 0:
                # 阻塞接收
                dist.recv(x_list[i], src = rank-1)
                print(f'[ recv] rank:{rank} <- rank:{rank-1} , micro_{i}')

            x_list[i].retain_grad()
            stage_output = pipe_model(x_list[i])
            stage_output_list[i] = stage_output

            if rank != world_size - 1:
                # 异步发送
                req = dist.isend(stage_output.clone(), dst = rank+1)
                reqs.append(req)
                print(f'[isend] rank:{rank} -> rank:{rank+1}, micro_{i}')

            if i ==  world_size - 1:
                '''
                参考 `lecture/lc1_basic/p2p_async.py` 异步通信实现
                '''
                if rank == 0:
                    for req in reqs:
                        req.wait()
                print(f'----rank_{rank}: finish isend----')
        dist.barrier()
        print('end forward')
        # if rank == world_size - 1:
            # print(stage_output_list)

        
        # backward
        '''
        反向时由于是micro batch, 需要实现梯度累积, 梯度累积每次算backward都会叠加,
        而执行optimizer.step()才完成参数更新
        '''
        if rank == 0:
            print('----pipeline backward---')
            
        reqs = []
        for i in range(world_size - 1, -1, -1): # reverse
            if rank != world_size - 1:
                dist.recv(stage_output_grad_list[i], src = rank + 1)
                print(f'[ recv] rank:{rank} <- rank:{rank+1} , micro_{i}')

            if rank == world_size - 1:
                loss = loss_fn(stage_output_list[i], label_list[i]) 
                loss /= world_size # 梯度累积
                loss.backward()
                print(loss)
            else:
                stage_output_list[i].backward(gradient = stage_output_grad_list[i])
                
            if rank != 0:
                req = dist.isend(x_list[i].grad.clone(), dst = rank-1)
                reqs.append(req)
                print(f'[isend] rank:{rank} -> rank:{rank-1}, micro_{i}')

            if i == 0:
                if rank == 0:
                    for req in reqs:
                        req.wait()
                print(f'----rank_{rank}: finish isend----')
        dist.barrier()      
        print('end backward')  
        optimizer.step()  
            
           
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
