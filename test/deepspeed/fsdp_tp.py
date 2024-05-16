#torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/fsdp_tp.py

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from typing import Callable
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.distributed._tensor import DeviceMesh, Shard, DTensor, Placement
from torch.distributed.fsdp.api import ShardingStrategy

fs_dim = 8
tp_dim = 2
input_dim = 2

class TPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn1 = torch.nn.Parameter(torch.empty(
            fs_dim,
            tp_dim,
        ))
    def forward(self, x):
        return self.ffn1 @ x
    def fsdp_wrap_fn(self, fsdp_mesh):
        return {'device_mesh': fsdp_mesh}

class FullShardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(input_dim, fs_dim, bias=False)
        self.net2 = nn.Linear(fs_dim, fs_dim, bias=False)
        self.ffn = TPModel()
    def forward(self, x):
        return self.ffn(self.net2(self.net1(x)))


def run_fsdp_checkpoint_save_example(rank, world_size, mesh_shape):
    ## INITIALIZE DIST
    # Running on one node so master_addr is just local host
    # os.environ["MASTER_ADDR"] = "localhost"

    
    # os.environ["MASTER_PORT"] = "28000"
    # All ranks simulataneously init the process group together.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    ## SETUP MODEL
    mesh_2d = init_device_mesh(
        'cuda', mesh_shape, mesh_dim_names=("dp", "tp")
    )
    tp_mesh = mesh_2d["tp"]
    ffn_fsdp_mesh = mesh_2d["dp"]
    mesh_1d = init_device_mesh('cuda', (world_size,))
    model = FullShardModel().to(rank)
    def dtensorify_param(param: nn.Parameter, mesh: DeviceMesh, placements: list[Placement]):
        """Construct a DTensor from an already sharded local parameter."""
        param_dtensor = DTensor.from_local(
            param.data,
            device_mesh=mesh,
            placements=placements,
            run_check=False,
        )
        return nn.Parameter(param_dtensor)
    dtensorified_params = [
        (
            name,
            dtensorify_param(
                param=parameter,
                mesh=tp_mesh,
                placements=[Shard(0)],
            )
        )
        for name, parameter in model.ffn.named_parameters()
    ]
    for name, dtensorified_param in dtensorified_params:
        model.ffn.register_parameter(name, dtensorified_param)


    ## ADD DTENSOR HOOKS
    def tensor_to_dtensor_hook(mod, args):
        inp, = args
        return DTensor.from_local(inp, tp_mesh, [Shard(0)])
    model.ffn.register_forward_pre_hook(tensor_to_dtensor_hook)

    def dtensor_to_tensor_hook(mod, inp, outp):
        return outp.to_local()
    model.ffn.register_forward_hook(dtensor_to_tensor_hook)

    ## WRAP MODEL
    def lambda_fn(module: torch.nn.Module):
        ret = False
        if hasattr(module, '_fsdp_wrap'):
            ret = bool(module._fsdp_wrap)
        elif hasattr(module, 'fsdp_wrap_fn') and isinstance(module.fsdp_wrap_fn, Callable):
            ret = module.fsdp_wrap_fn(ffn_fsdp_mesh)
        return ret

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_mesh=mesh_1d,
        use_orig_params=True,
        auto_wrap_policy=CustomPolicy(lambda_fn),
    )
    print(model)

    ## RUN MODEL
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    optim.zero_grad()
    outp = model(torch.rand(4,input_dim,device="cuda")).sum(dim=1)
    loss = loss_fn(outp, torch.ones(8, device='cuda'))
    loss.backward()
    optim.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    mesh_shape = (1, 2)
    world_size = mesh_shape[0] * mesh_shape[1]
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args = (world_size, mesh_shape),
        nprocs=world_size,
        join=True
    )
