# python lm_head.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.autograd as autograd
from col_parallel_linear import ColParallelLinear, ColFunction

class CrossEntropyFunction(autograd.Function):
    '''
    推导参考: [cut-cross-entropy](https://zhuanlan.zhihu.com/p/13548439339)
    E : [BS, N, D]
    C : [D, V/4], 为输出投影矩阵, 可见输出为V/4词表大小, 存在求Softmax一致性问题
    Y : [BS, N], 这里的 Y 为全词表label, 由于局部词表范围固定, 所以要过滤非本词表的 label
    '''
    @staticmethod
    def forward(ctx, E, C, Y, rank):
        d, local_vocab_size = C.shape

        logits = E @ C # b x n x d   d x v/4 -> n x v/4
        LSE = torch.log(torch.sum(logits.exp(), dim = -1, keepdim=True))

        # LSE =  [exp(LSE1) + exp(LSE2)].log()
        LSE = LSE.exp()
        dist.all_reduce(LSE, dist.ReduceOp.SUM)
        LSE = LSE.log()

        # log(softmax(x)) = x - LSE
        log_prob_all = logits - LSE

        if Y != None:
            local_Y_masked = torch.logical_and( Y >= rank*local_vocab_size, Y < (rank+1) * local_vocab_size)
            Y_valid_vocab = Y.clone()
            idx = torch.where(local_Y_masked == False)
            Y_valid_vocab[idx] = (rank * local_vocab_size)

            gather_idx = Y_valid_vocab - int(rank * local_vocab_size)

            log_prob = torch.gather(log_prob_all, dim=2, index = gather_idx.unsqueeze(-1)).squeeze(-1)

            loss = -(log_prob * local_Y_masked).sum() / local_Y_masked.sum()

            ctx.save_for_backward(E, C,)
            ctx.custom_obj = [rank, Y, LSE]
            return loss, logits, log_prob_all
        else:
            return None, logits, log_prob_all
    
    @staticmethod
    def backward(ctx, grad_output, a, b):
        E, C,  = ctx.saved_tensors
        rank, Y, LSE = ctx.custom_obj
        _, vocab = C.shape

        local_Y_masked = torch.logical_and( Y >= rank*vocab, Y < (rank+1) * vocab)
        Y_valid_vocab = Y.clone()
        idx = torch.where(local_Y_masked == False)
        # Y_valid_vocab[idx] = 0
        Y_valid_vocab[idx] = (rank * vocab)
        
        one_hot_label = nn.functional.one_hot(Y_valid_vocab - (rank * vocab), 
                                              num_classes = vocab)
        
        one_hot_label = one_hot_label * local_Y_masked.unsqueeze(-1).repeat(1,1,vocab)


        prob = LSE.exp() - E @ C # bs, n, v/4,   v/4
        grad_input = grad_output * (prob - one_hot_label) @ C.t()
        grad_C = grad_output * (E.transpose(2,1) @ (prob - one_hot_label)).sum(dim=0) # bxnxd bxnxv/4

        dist.all_reduce(grad_input, dist.ReduceOp.SUM)
        return  grad_input, grad_C, None, None
    
class LanguageModelHead(nn.Module):
    def __init__(self, dim, vocab_size, rank = 0, world_size = 1):
        super(LanguageModelHead, self).__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.rank = rank
        self.world_size = world_size
        self.lm_head = ColParallelLinear(dim, vocab_size, self.rank, self.world_size) 
    
    def forward(self, x, y):
        # 单卡实现
        # logits = self.lm_head(x)
        # log_prob = nn.functional.log_softmax(logits, dim = -1)

        # 分布式版本
        loss, logits, log_prob= CrossEntropyFunction.apply(x, self.lm_head.w.weight.t(), y, self.rank)
        bs, seq_len, _ = x.shape
        
        logits_gather_list = [torch.zeros(bs, seq_len, self.vocab_size // self.world_size) 
                              for _ in range(self.world_size)]
        log_prob_list = [torch.zeros(bs, seq_len, self.vocab_size // self.world_size) 
                    for _ in range(self.world_size)]
        
        dist.all_gather(logits_gather_list, logits)
        dist.all_gather(log_prob_list, log_prob)

        logits_gather = torch.concat(logits_gather_list, dim = -1)
        log_prob_list = torch.concat(log_prob_list, dim = -1)

        return loss, logits_gather, log_prob_list


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    lm_head的投影矩阵随着词表大小线性增大, 本代码实现在vocab维度进行切分, 减少少数矩阵规模
    实现难点在于如何融合CrossEntropy来实现梯度更新
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:' + master_port,
                            rank=rank, 
                            world_size=world_size)

    bs = 16
    seq_len = 32
    dim = 64
    vocab_size = 512

    label = torch.randint(size=[bs, seq_len], high=vocab_size) # 4x32   32 x 16
    input = torch.randn(bs, seq_len, dim, requires_grad=True)
    dist.broadcast(input, src = 0)
    dist.broadcast(label, src = 0)

    model = LanguageModelHead(dim, vocab_size, rank, world_size)
    loss, logits_gather, log_prob_list, = model(input, label)

    if rank == 0:
        print(loss)

    loss.backward()

    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 8, ), nprocs=8)