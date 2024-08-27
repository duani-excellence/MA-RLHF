# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_tp_colwise.py

import os
import torch
import torch.distributed as dist


def is_main_device():
    rank = dist.get_rank()
    return rank == 0


def setup():
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def dprint(message):
    rank = dist.get_rank()
    print(f"[Rank {rank}] {message}")


class ColWisePararrel():
    def __init__(self, dim=6, dim_out=10, n_gpus=2, device_id=0) -> None:
        self.dim = dim
        self.dim_out = dim_out
        self.n_gpus = n_gpus
        self.col_per_device = dim_out // n_gpus
        self.w = torch.zeros(dim, self.col_per_device).to(device_id)
        dprint(self.w.shape)
        self.lr = 0.01

    def scatter_w(self, device_id):
        if is_main_device():
            w_all = torch.randn(self.dim, self.dim_out).to(device_id)
            w_list = torch.split(w_all, self.col_per_device, dim=1)
            w_list = list(w_list)
        else:
            w_list = None
        dist.scatter(tensor=self.w, scatter_list=w_list, src=0)
        dist.barrier()

    def forward(self, x):
        y = x @ self.w
        return y

    def backward(self, x, error):
        grad = x.t() @ error * (1/self.dim_out)
        return grad

    def update(self, grad):
        self.w -= self.lr * grad

    def gather_w(self, device_id):
        # w = torch.zeros(self.dim, self.col_per_device).to(device_id)
        if is_main_device():
            w_list = [torch.zeros_like(self.w) for _ in range(self.n_gpus)]
        else:
            w_list = None
        dist.gather(self.w, w_list, dst=0)
        return w_list


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    setup()
    rank = dist.get_rank()
    n_gpus = torch.cuda.device_count()
    pid = os.getpid()
    print(f'current pid: {pid}')
    print(f'Current rank {rank}')

    in_dim = 6
    out_dim = 10
    bs = 2
    x = torch.randn(bs, in_dim).to(rank)
    y_label = torch.zeros(bs, out_dim//n_gpus).to(rank)

    # ----------------------------------------------------------------------------
    dprint('----col parallelism start------')
    if is_main_device():
        y_labels = torch.randn(bs, out_dim).to(rank)
        # torch split return TUPLE, NOT LIST!!
        y_labels = torch.split(y_labels, out_dim//n_gpus, dim=1)
        y_labels = list(y_labels)
    else:
        y_labels = None
    dist.scatter(tensor=y_label, scatter_list=y_labels, src=0)
    dist.barrier()

    model = ColWisePararrel(dim=6, dim_out=10, n_gpus=n_gpus, device_id=rank)
    model.scatter_w(rank)
    x = torch.randn(bs, in_dim).to(rank)
    y = model.forward(x)
    e = y - y_label
    grad = model.backward(x, e)
    model.update(grad)
    w_dist = model.gather_w(rank)

    if is_main_device():
       w = torch.cat(w_dist, dim=1)
       dprint(w.shape)
    dprint('----col parallelism end------')
    cleanup()


if __name__ == '__main__':
    main()
