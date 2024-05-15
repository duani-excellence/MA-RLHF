# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_profiler.py

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class TwoLinLayerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10, bias=False)
        self.b = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    print("init model")
    model = TwoLinLayerNet().cuda()
    print("init ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    inp = torch.randn(10, 10).cuda()
    print("train")

    for _ in range(20):
        output = ddp_model(inp)
        loss = output[0] + output[1]
        loss.sum().backward()


if __name__ == "__main__":
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    mp.spawn(worker, nprocs=2, args=())
