# python ring_attention.py 
'''
以下代码实现了 RingAttention 的前向和后向计算
1. 每个rank的参数是一致的
2. Attention的计算参考 Flash Attention-V2实现
3. RingAttention 天然是Block-Wise的注意力场景, 每个 rank 的Q块固定, 而 KV 块环形传输
4. 如果加入 RoPE, 需要注意块KV的位置需要转化为绝对位置
5. 从GQA 和 MLA角度思考, KV块环型传输量更小

从Infrence Prefill过程, 则可以复用 Ring Attention 前向
'''

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
import math
torch.manual_seed(42)

class RingAttention:
    def __init__(self, dim = 512, heads = 8, vocab_size = 100, rank = 0, world_size = 1):
        super(RingAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.vocab_size = vocab_size
        self.head_dim = self.dim // self.heads
        self.WQ = nn.Linear(self.dim, self.dim, bias=False)
        self.WK = nn.Linear(self.dim, self.dim, bias=False)
        self.WV = nn.Linear(self.dim, self.dim, bias=False)
        self.WO = nn.Linear(self.dim, self.dim, bias=False)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.rank = rank
        self.world_size = world_size
    
    def share_parameters(self):
        '''
        不同rank参数保持一致
        '''
        dist.broadcast(self.WQ.weight.clone(), src = 0)
        dist.broadcast(self.WK.weight.clone(), src = 0)
        dist.broadcast(self.WV.weight.clone(), src = 0)
        dist.broadcast(self.WO.weight.clone(), src = 0)
    
    def proj(self, X):
        bs, seq, _ = X.shape
        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
        Q = Q.view(bs, seq, self.heads, self.dim//self.heads).transpose(1,2).contiguous()
        K = K.view(bs, seq, self.heads, self.dim//self.heads).transpose(1,2).contiguous()
        V = V.view(bs, seq, self.heads, self.dim//self.heads).transpose(1,2).contiguous()
        return Q, K, V
    
    def block_attention_forward(self, Q, K, V, L, M, O):
        _, _, _, head_dim = Q.shape
        S = Q @ K.transpose(3,2) / math.sqrt(head_dim)
        M_local = torch.max(S, dim = -1, keepdim = True).values
        M_new = torch.maximum(M, M_local)
        L_local = torch.sum(torch.exp(S - M_new) , dim = -1, keepdim = True)
        L_new = L * torch.exp(M - M_new) + L_local
        O_new =  O * torch.exp(M - M_new) + torch.exp(S - M_new) @ V
        return L_new, M_new, O_new

    def step_forward(self, Q, K, V):
        '''
        1. 输入数据Q, K, V 本身已经是块数据了
        2. q_len 不一定与 k_len 相同
        3. ring传播的是KV
        '''
        bs, q_head, q_len, head_dim = Q.shape
        bs, k_head, k_len, head_dim = K.shape

        assert q_head == k_head, 'not same head'

        L = torch.zeros(bs, q_head, q_len, 1)
        M = torch.ones(bs, q_head, q_len, 1) * -10000.0
        O = torch.zeros(bs, q_head, q_len, head_dim)

        for i in range(self.world_size):
            # 先计算, 执行Flash Attention V2
            L, M, O = self.block_attention_forward(Q, K, V, L, M, O)
            # 再传播数据
            K, V = self.ring_comm_KV(K, V)
        O = O / L # scaled
        L_b = M + L.log() # 用于反向过程重计算softmax

        return O, L_b

    def block_attention_backward(self, Q, K, V, L_b, O, dO, D):
        _, _, _, head_dim = Q.shape
        dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V), 
        S = Q @ K.transpose(3,2) / math.sqrt(head_dim)
        P = S - L_b
        # print(S.shape)
        # print(L_b.shape)
        # print(Q.shape)
        # print(K.shape)
        # print(P.shape)
        # print(dO.shape)
        dV += P.transpose(3,2) @ dO
        dP = dO @ V.transpose(3,2)
        dS = P * (dP - D)
        dQ = dS @ K / math.sqrt(head_dim)
        dK = dS.transpose(3,2) @ Q / math.sqrt(head_dim)
        return dQ, dK, dV

    def step_backward(self, Q, K, V, L_b, O, dO):

        bs, q_head, q_len, head_dim = Q.shape
        bs, k_head, k_len, head_dim = K.shape

        assert q_head == k_head, 'not same head'

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        D = torch.sum(O * dO, dim = -1, keepdim=True)
        for i in range(self.world_size):
            # 先计算, 执行Flash Attention V2
            dQ_block, dK_block, dV_block = self.block_attention_backward(Q, K, V, L_b, O, dO, D)
            dQ += dQ_block
            dK += dK_block
            dV += dV_block
            # 再传播数据
            K, V = self.ring_comm_KV(K, V)
        return dQ, dK, dV

    def step_backward_attention(self, Q, K, V):
        return

    def ring_comm_KV(self, K, V):
        tmp_K = torch.zeros_like(K)
        tmp_V = torch.zeros_like(V)

        next_rank = (self.rank + 1) %  self.world_size
        pre_rank = (self.rank - 1) %  self.world_size

        if self.rank % 2 == 0:
            dist.send(K, dst = next_rank)
            dist.recv(tmp_K, src = pre_rank)
        else:
            dist.recv(tmp_K, src = pre_rank)
            dist.send(K, dst = next_rank)
        dist.barrier()

        if self.rank % 2 == 0:
            dist.send(V, dst = next_rank)
            dist.recv(tmp_V, src = pre_rank)
        else:
            dist.recv(tmp_V, src = pre_rank)
            dist.send(V, dst = next_rank)
        dist.barrier()

        return tmp_K, tmp_V

    def loss_fn(self, O, Y):
        logits = self.WO(O)
        loss = (logits - Y) ** 2 / O.numel()
        return logits, loss
    

    def loss_fn_grad(self, logits, Y):
        loss = 2 * (logits - Y) / logits.numel()
        return loss
    
    def all_reduce_avg(self, W):
        dist.all_reduce(W, dist.ReduceOp.SUM)
        W /= self.world_size
        return W

    def backward_gradient_update(self, X, dQ, dK, dV, O, dO):
        bs, seq, dim = X.shape

        def proj_backward(dQ, X, WQ):
            dQ_linear = dQ.transpose(2,1).reshape(bs, seq, dim)
            dW = dQ_linear.transpose(2,1) @ X
            dW = dW.sum(dim = 0)
            dX = dQ_linear @ WQ
            return dW, dX

        d_WQ, dX_Q = proj_backward(dQ, X, self.WQ.weight)
        d_WK, dX_K = proj_backward(dK, X, self.WK.weight)
        d_WV, dX_V = proj_backward(dV, X, self.WV.weight)
        d_WO, _ = proj_backward(dO, O, self.WO.weight)

        self.WQ.weight.grad = self.all_reduce_avg(d_WQ)
        self.WK.weight.grad = self.all_reduce_avg(d_WK)
        self.WV.weight.grad = self.all_reduce_avg(d_WV)
        self.WO.weight.grad = self.all_reduce_avg(d_WO)

        dX = dX_Q + dX_K + dX_V

        return dX
    
    def step(self, X, Y = None):
        '''
        X是已经被切分了的tensor [BS, sub_seq, dim], 实现步骤为:
        1. 投影 输入部分X rank1: [Q1,Q2, K1,K2, V1,V2], rank2: [Q3,Q4, K3,K4, V3,V4]
        2. 前向
        3. 后向
        '''
        bs, sub_seq, dim = X.shape

        # step1: proj
        Q, K, V = self.proj(X) # Q: [BS, sub_seq, head, dim]
        # print(Q.shape, K.shape, V.shape)
        # return None

        # # step2: forward
        O, L_b = self.step_forward(Q, K, V)
        O_cat = O.clone().view(bs, sub_seq, dim)
        # return None, None

        # step3: backward
        logits = None
        if Y != None:
            logits, loss = self.loss_fn(O_cat, Y)
            dE = self.loss_fn_grad(logits, Y)
            dO = dE @ self.WO.weight
            dO = dO.view(bs, sub_seq, self.heads, self.head_dim).transpose(2,1).contiguous()
            dQ, dK, dV = self.step_backward(Q, K, V, L_b, O, dO)

            # step4: update
            # update WQ, WK, WV, WO
            dX = self.backward_gradient_update(X, dQ, dK, dV, O_cat, dO)
            # dX backward 
        return loss, logits
        

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    实现增量式的online softmax
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)
    bs = 1
    seq_len = 32
    dim = 512
    heads = 8
    vocab_size = 100 # ignore
    input = torch.randn(bs, seq_len, dim)
    label = torch.randn(bs, seq_len, dim)
    input_block = torch.zeros(bs, seq_len // world_size, dim)
    label_block = torch.zeros(bs, seq_len // world_size, dim)
    if rank == 0:
        input_block_list = list(input.chunk(world_size, dim = 1))
        label_block_list = list(label.chunk(world_size, dim = 1))
    else:
        input_block_list = []
        label_block_list = []
    dist.scatter(input_block, input_block_list, src = 0)
    dist.scatter(label_block, label_block_list, src = 0)

    model = RingAttention(dim, heads, vocab_size, rank, world_size)
    model.share_parameters()

    # for i in range( epochs ):
    #   optimizer.zero_grad()
    if torch.no_grad():
        loss, logits = model.step(input_block, label_block)
        # loss = model.all_reduce_avg(loss)
        if rank == 0:
            print(loss)
        # optimizer.step()

    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)


