# python pipeline_parallel_zero_bubble.py
'''
GPipe: N-F-N-B
PipeDream: 1-F-1-B
ZeroBubble: 1-F-1-B-1-W, W为权重梯度更新步骤.
ZB将反向传播时对输入求导和对参数求导分离, 目的是实现 “通信-计算重叠”
    - t1: backward时先计算对输入的求导 dx, 
    - t2: 通信: 将 dx 传递给上一层(异步isend)
    - t2: 计算: 将 dw 进行计算
实现的难点在于:
    -1. 如何在计算图上分离对输入求导和参数求导的顺序
    -2. 在t1先对dx求导过程中, 仍有中间的梯度进行保存
            -----------------
            forward:
            - y1 = x1 * w1
            - y2 = y1 * w2
            -----------------
            backward(x):
            - dy1 = dy2 * w2
            - dx1 = dy1 * w1
            save(dy1, dx1)   # 增加显存占用
            comm~
            -----------------
            backward(w):
            - dw2 = dy2 * y1
            - dw1 = dy1 * w1
            ----------------
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.autograd as autograd
import torch.multiprocessing as mp
import copy


class ZeroBubbleMLP(nn.Module):
    def __init__(self, dim, rank = 0, world_size = 1):
        super(ZeroBubbleMLP, self).__init__()
        self.dim = dim
        self.rank = rank
        self.world_size = world_size
        self.w1 = nn.Linear(dim, dim * 4, bias = False)
        self.w2 = nn.Linear(dim * 4, dim, bias = False) # data shape, dim x 4dim
    
    def forward(self, x):
        h = self.w1(x) 
        o = self.w2(h)
        return h, o 
    
    def loss_backward(self, o, label):
        do = 2 * (o - label) / o.numel()
        return do

    def backward_for_input(self, do):
        '''
        '''
        dh = do @ self.w2.weight # 丢弃
        dx = dh @ self.w1.weight
        return dh, dx

    def backward_for_weight(self, do, dh, h, x):
        '''
        '''
        self.w2.weight.grad = do.t() @ h # dim x bs @ bs x 4dim -> dim x 4dim
        self.w1.weight.grad = dh.t() @ x 
        return None


class ZeroBubbleModel(nn.Module):
    def __init__(self, dim, num_blocks, rank = 0, world_size = 1):
        super(ZeroBubbleModel, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.dim = dim
        self.num_blocks = num_blocks
        self.layers = torch.nn.ModuleList()
        self.local_num_blocks = num_blocks // world_size
        for i in range(self.local_num_blocks):
            self.layers.append(ZeroBubbleMLP(self.dim, self.rank, self.world_size))
        
    def forward(self, x): 
        '''
        return [[x1,h1], [x2, h2]] |  x3
        '''
        layers_output = [ [] for i in range(self.local_num_blocks)]
        for i, layer in enumerate(self.layers):
            layers_output[i].append(x)
            h, x = layer(x)
            layers_output[i].append(h)
        return layers_output, x # [[input, hidden],...], output
    
    def backward_zero_bubble(self, layers_output, do, b_idx, is_send = True, dst= None):
        # layers_output_for_weight = deepcopy.copy(layer_output)
        dx = None
        for layer, layer_output in zip(reversed( self.layers), reversed(layers_output)):
            dh, dx = layer.backward_for_input(do)
            layer_output.append(dh)
            layer_output.append(dx)

        # 将 dx 传送到上一个 rank
        if self.rank != 0 and is_send:
            if dst == None:
                req = dist.isend(dx, dst = self.rank-1, tag=10086)
                print(f'[rank{self.rank}] micro_batch:{self.world_size-b_idx-1}, dx isend-backward')
            else:
                req = dist.isend(dx, dst = dst, tag=10086)
                print(f'[rank{self.rank}] dst:{dst}, dx isend-backward')
            req.wait()
        # 传送完后需要把数据清除掉
        
        # 计算梯度
        for layer, layer_output in zip(reversed( self.layers), reversed(layers_output)):
            layer.backward_for_weight(
                x = layer_output[0],
                h = layer_output[1],
                dh = layer_output[2],
                do = do,
            )
        # print(f'[rank{self.rank}] micro_batch:{b_idx}, dw ')
        return dx

        

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    # 准备数据
    dim = 512
    num_blocks = 8
    bs = 128
    micro_batch_size = world_size # 8条数据 4个rank


    if rank == 0:
        x = torch.randn(bs, dim, requires_grad=True)
    else:
        x = torch.zeros(bs, dim, requires_grad=True)
    x_list = list(torch.chunk(x, micro_batch_size, dim = 0))

    if rank == world_size-1:
        label = torch.randn(bs, dim).sin()
        label_list =  list(torch.chunk(label, micro_batch_size, dim = 0))
        loss_fn = nn.MSELoss()

    stage_output = torch.zeros_like(x_list[0], requires_grad=True)
    stage_output_grad = torch.zeros_like(x, requires_grad=True)

    # 注意中间变量的存储不等同于micro_batch_size
    # 即存储的中间变量是有限的
    stage_output_list = [torch.zeros_like(x_list[0]) for _ in range(world_size)]
    stage_output_grad_list = [torch.zeros_like(x_list[0]) for _ in range(world_size)]
    stage_output_tmp_list = [None] * 4
    

    pipe_model = ZeroBubbleModel(dim, num_blocks=num_blocks, rank = rank, world_size=world_size)
    optimizer = optim.SGD(pipe_model.parameters(), lr = 0.0001)

    # zero_bubble 分离dx和dw的计算

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

        with torch.no_grad():
            for i in range(micro_batch_size + world_size - 1):
                cur_f_idx = f_idx % world_size
                if i >= rank and f_idx < micro_batch_size: 
                    if rank != 0:
                        # 阻塞接收
                        dist.recv(x_list[f_idx], src = rank-1, tag = 10010)
                        print(f'\t [cur_rank:{rank}], it:{i} - [1F-recv] rank:{rank} <- rank:{rank-1} , micro_{f_idx}')

                    x_list[f_idx].retain_grad()
                    stage_output_tmp, stage_output = pipe_model(x_list[f_idx])
                    stage_output_list[cur_f_idx] = stage_output
                    stage_output_tmp_list[cur_f_idx] = stage_output_tmp
                    
                    it_log.append('F'+ str(f_idx))

                    if rank != world_size - 1:
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
                    cur_b_idx = b_idx % world_size

                    if rank != world_size - 1 :
                        dist.recv(stage_output_grad_list[cur_b_idx], src = rank + 1, tag = 10086)
                        print(f'\t [cur_rank:{rank}], it:{i} - [1B-recv] rank:{rank} <- rank:{rank+1} , micro_{b_idx}')

                    if rank == world_size - 1:
                        do = pipe_model.layers[-1].loss_backward(stage_output_list[cur_b_idx], label_list[b_idx])
                    else:
                        do = stage_output_grad_list[cur_b_idx]
                    pipe_model.backward_zero_bubble(stage_output_tmp_list[b_idx], do, b_idx)
                    print(f'[rank{rank}] micro_batch:{b_idx}, mid-layer backward')

                    it_log.append('B' + str(b_idx))
                    it_log.append('w' + str(b_idx))
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
