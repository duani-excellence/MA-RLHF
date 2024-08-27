
# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_tp_rowwise.py

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

class RowWisePararrel():
    def __init__(self, dim=6, dim_out=10, n_gpus=2, device_id = 0) -> None:
        self.dim = dim
        self.dim_out = dim_out
        self.n_gpus = n_gpus
        self.row_per_device = dim // n_gpus
        self.w = torch.ones(self.row_per_device, dim_out).to(device_id)
        self.lr = 0.01
        if is_main_device():
            w = torch.randn(dim, dim_out, requires_grad = True)
            w_list = torch.split(w, self.row_per_device, dim = 1)
        dist.scatter(self.w, w_list, src = 0)

    def forward(self, x):
        y = self.w(x)
        return y

    def backward(self, x, error):
        grad = x.t() @ error * (1/self.dim_out)
        return grad

    def update(self, grad):
        self.w -= self.lr * grad

    def gather_w(self, ):
        w = torch.zeros(self.dim, self.dim_out)
        if is_main_device():
            w_list = [torch.zeros_like(self.w) for _ in range(self.n_gpus)]
        dist.gather(w, w_list, dst = 0)
        return w


def main():
    setup()
    rank = dist.get_rank()
    n_gpus = torch.cuda.device_count()
    pid = os.getpid()
    print(f'current pid: {pid}')
    print(f'Current rank {rank}')
    device_id = rank % torch.cuda.device_count()

    in_dim = 6
    out_dim = 10
    bs = 2
    x = torch.randn(bs, in_dim//2).to(rank)
    y_label = torch.randn(bs, out_dim).to(rank)


    # ----------------------------------------------------------------------------
    dprint('----row parallelism start------')
    if is_main_device():
        x_all = torch.randn(bs, in_dim)
        x_list = torch.split(x_all, in_dim//n_gpus, dim=1)
    dist.scatter(x_list, x, src=0)

    model = RowWisePararrel(dim=6, out_dim=10, n_gpus=n_gpus, device_id=rank)


    y = model.forward(x)
    # y_result = torch.zeros_like(y).to(rank)

    if is_main_device():
        y_list = [torch.zeros_like(y) for _ in range(n_gpus)]
    dist.all_reduce(y, op=dist.ReduceOp.SUM)

    e = y - y_label
    grad = model.backward(x, e)
    model.update(x, grad)
    w_dist = model.gather_w()

    if is_main_device():
       w = torch.cat(w_dist, dim=0)
       print(w.shape)
    dprint('----row parallelism end------')


if __name__ == '__main__':
    main()
