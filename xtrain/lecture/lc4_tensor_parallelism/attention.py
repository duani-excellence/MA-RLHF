# python attention.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd

from row_parallel_linear import RowParallelLinear, RowFunction
from col_parallel_linear import ColParallelLinear, ColFunction


class AttentionFunction(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, KV_src, group):
        S = Q @ K.transpose(1, 2)
        P = torch.nn.functional.softmax(S, dim=-1)
        Z = P @ V
        ctx.save_for_backward(Q, K, V, P, S)
        ctx.custom_obj = [KV_src, group]
        return Z

    @staticmethod
    def backward(ctx, dZ):
        Q, K, V, P, S = ctx.saved_tensors
        KV_src, group = ctx.custom_obj

        dP = dZ @ V.transpose(1, 2)
        dV = P.transpose(1, 2) @ dZ

        dS = torch.zeros_like(P)

        for b in range(Q.shape[0]):
            for i in range(Q.shape[1]):
                I = torch.diag(P[b, i, :]) - torch.outer(P[b, i, :],
                                                         P[b, i, :])
                dS[b, i, :] = dP[b, i, :] @ I

        # dQ = K.transpose(1, 2) @ dS
        dQ = dS.transpose(1, 2) @ K
        dK = dS.transpose(1, 2) @ Q

        # 1~4 卡各自有 dk, 那么需要 all-reduce
        #  因为当初的 k = x wk,  k 发放到了 1～4 卡中
        #  dk   ｜  dk  ｜  dk  ｜  dk  ｜
        #   \       |       |      /
        #     \      \     /     /
        #            sum(dk)
        #               |
        #               V
        #           wk <- wk + dk
        dist.all_reduce(dK,
                        dist.ReduceOp.SUM,
                        group)

        dist.all_reduce(dV,
                        dist.ReduceOp.SUM,
                        group)

        return dQ, dK, dV, None, None


class Attention(nn.Module):
    '''
    1. 张量并行里, 由于不同头是特征维度上是独立的, 用列并行来实现变换.
    2. 我们直接实现 GQA 版本, 这里要注意组内的KV是一致的.
        假设 8卡, 8头Q, 2头KV, 则卡GPU0,1,2,3 用【K1V1】和 GPU4,5,6,7 【K2V2】
    3. 在反向时要注意WKV的梯度的reduce
    4. 注意力计算完后要通过 wo 的变换, 则将 wo 参数设置成行并行
    '''

    def __init__(self, dim, n_kv_heads, heads, rank=0, world_size=1):
        super(Attention, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.head_dim = dim // heads
        self.heads = heads
        self.n_kv_heads = n_kv_heads

        # GPU 1~8 共享
        self.WQ = ColParallelLinear(dim, (self.head_dim * heads),
                                    rank, world_size)
        # GPU 1~4 共享, 5~8 共享参数
        # 思考: 按照分布式版本的 GQA 实现, 推理的 KV-Cache 如何来存储？
        #     a. 组内各卡存储重复的 KV-Cache
        #     b. 将一份 KV-Cache 散步在组内多个卡中
        #     c. 将组 GPU 显存统一管理, 并 page-式 管理 KV-Cache
        self.WK = ColParallelLinear(dim, (self.head_dim * heads),
                                    rank, world_size)
        self.WV = ColParallelLinear(dim, (self.head_dim * heads),
                                    rank, world_size)

        # GQA创建 n_kv_heads 个组
        # 每组共享头数量为 heads // n_kv_heads
        gpus = list(range(0, world_size))
        self.shared_heads = heads // n_kv_heads  # 一个kv头, 共享Q头数量
        ranks_groups = [gpus[i:i+self.shared_heads]
                        for i in range(0, len(gpus), self.shared_heads)]
        
        # print('rank', self.rank, 'GQA groups: ',ranks_groups)

        self.cur_group = self.rank // self.shared_heads  # [4,4]
        self.kv_rank_src = self.cur_group * self.shared_heads
        self.gqa_groups = [dist.new_group(ranks=ranks)
                           for ranks in ranks_groups]

        dist.broadcast(self.WK.w.weight.data,
                       src=self.kv_rank_src,
                       group=self.gqa_groups[self.cur_group])
        dist.broadcast(self.WV.w.weight.data,
                       src=self.kv_rank_src,
                       group=self.gqa_groups[self.cur_group])

        # wo 初始化
        self.WO = RowParallelLinear(
            dim, (self.head_dim * heads), rank, world_size)

    def forward(self, x):
        Q, K, V = self.WQ(x), self.WK(x), self.WV(x)
        Z = AttentionFunction.apply(
            Q, K, V, self.kv_rank_src, self.gqa_groups[self.cur_group])

        O = RowFunction.apply(Z, self.WO.w.weight.t())
        # dist.all_reduce(O, dist.ReduceOp.SUM)
        return O


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    mlp 实现, 先列并行, 再行并行
    X[N, d], W1[d, 4d], W2[4d, d]
    XW1' -> N, d;    W2'[d,d]
    [XW1'W2'] -> reduce([N, d])
    '''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank,
                            world_size=world_size)

    N = 32
    seq_len = 5
    dim_in = 64
    label = torch.zeros(N, seq_len, dim_in)

    if rank == 0:
        x = torch.randn(N, seq_len, dim_in)
    else:
        x = torch.zeros(N, seq_len, dim_in)
    dist.broadcast(x, src=0)
    dist.barrier()

    # GQA
    model = Attention(dim=dim_in,
                      n_kv_heads=2,
                      heads=world_size,
                      rank=rank,
                      world_size=world_size)

    y = model(x)

    loss_fn = nn.MSELoss()
    loss = loss_fn(y, label)
    loss.backward()

    print(loss)

    dist.destroy_process_group()


if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)
