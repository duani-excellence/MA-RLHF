# python pipeline_parallel_pipe_dream.py
# GPipe 在一批数据里完成了“所有的”前向计算后, 才开始“所有的”后向计算
# rank1  Fa Fb  Fc  Fd
# rank2     Fa  Fb  Fc  Fd
# rank3         Fa  Fb  Fc  Fd
# rank4             Fa  Fb  Fc  [Fd  Bd] 
# GPipe 的问题在于需要存储过多的中间状态, 导致显存占用过多
#在rank4算完Fd数据后开始算反向, 需要存储Fa, Fb, Fc, Fd中间变量


# PipeDream 则思考如micro-batch里一个最块算出loss后直接就算backward, 示例如下:
# rank1  Fa Fb
# rank2     Fa  Fb
# rank3         Fa  Fb 
# rank4            [Fa  Ba] [Fb Bb]
# PipeDream 在rank4算完Fa数据后及时算反向, 则不需要存储Fa中间变量

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from pipeline_parallel_basic import PipeModel


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    # 准备数据
    dim = 512
    num_blocks = 8
    bs = 128
    micro_batch_size = world_size
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
    # stage_output_list = list(torch.chunk(stage_output, world_size, dim = 0))
    stage_output_list = []
    stage_output_grad_list = list(torch.chunk(stage_output_grad, world_size, dim = 0))
    

    pipe_model = PipeModel(dim, num_blocks=num_blocks)
    optimizer = optim.SGD(pipe_model.parameters(), lr = 0.0001)

    # Pipe Dream 实现思路为:
    # 1. 规定"总时序"(包含前向后向)为 2 * gpus - 1
    # 2. 时序 i >= rank 执行 forward
    # 3. 当rank 3进行backward时, 其他的GPU均是阻塞的

    for e in range(1):

        optimizer.zero_grad()
        stage_output.grad = None
        stage_output_grad.grad = None
        x.grad = None
        
        if rank == 0:
            print('----pipeline forward---')

        reqs_f = []
        reqs_b = []
        f_idx = 0
        b_idx = 0
        it_log = []

        for i in range(world_size + micro_batch_size - 1):
            if i >= rank and f_idx < micro_batch_size: 
                if rank != 0:
                    # 阻塞接收
                    dist.recv(x_list[f_idx], src = rank-1, tag = 10010)
                    print(f'\t [cur_rank:{rank}], it:{i} - [1F-recv] rank:{rank} <- rank:{rank-1} , micro_{f_idx}')

                x_list[f_idx].retain_grad()
                stage_output = pipe_model(x_list[f_idx])
                stage_output_list.append( stage_output)
                it_log.append('F'+ str(f_idx))

                if rank != micro_batch_size - 1:
                    # 异步发送
                    req = dist.isend(stage_output.clone(), dst = rank+1, tag = 10010)
                    reqs_f.append(req)
                    print(f'[cur_rank:{rank}], it:{i} - [1F-isend] rank:{rank} -> rank:{rank+1}, micro_{f_idx}')
                f_idx += 1
            else:
                # it_log.append('NF')
                it_log.append('--')

            # 1B
            if i >= world_size - 1 and b_idx < micro_batch_size:
                if rank != world_size - 1 :
                    dist.recv(stage_output_grad_list[b_idx], src = rank + 1, tag = 10086)
                    print(f'\t [cur_rank:{rank}], it:{i} - [1B-recv] rank:{rank} <- rank:{rank+1} , micro_{b_idx}')

                if rank == world_size - 1:
                    loss = loss_fn(stage_output_list[b_idx], label_list[b_idx]) 
                    loss /= world_size # 梯度累积
                    loss.backward()
                    print(f'[rank{rank}] micro_batch:{world_size-b_idx-1}, loss:{loss}')
                    it_log.append('B'+ str(b_idx))
                else:
                    stage_output_list[b_idx].backward(gradient = stage_output_grad_list[b_idx])
                    print(f'[rank{rank}] micro_batch:{world_size-b_idx-1}, mid-layer backward')
                    it_log.append('B'+ str(b_idx))

                if rank != 0 :
                    # print(f'[1B-isend] rank:{rank} -> rank:{rank-1}, micro_{b_idx+1}')
                    req = dist.isend(x_list[b_idx].grad.clone(), dst = rank-1, tag = 10086)
                    reqs_b.append(req)
                    print(f'[cur_rank:{rank}], it:{i} - [1B-isend] rank:{rank} -> rank:{rank-1}, micro_{b_idx}')
                b_idx += 1
            else:
                it_log.append('--')
                # it_log.append('NB')
        dist.barrier()      
        print('end backward')  
        # print(f'[rank-{rank}]', it_log)
        it_log_gather = [None] * world_size
        dist.all_gather_object(it_log_gather, it_log)
        if rank == 0:
            for cur_log in it_log_gather:
                print(cur_log)
        optimizer.step()  
        
            
    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)
