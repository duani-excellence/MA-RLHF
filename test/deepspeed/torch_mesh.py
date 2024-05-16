# torchrun  --nnodes=1 --nproc_per_node=2 ./test/deepspeed/torch_mesh.py

from torch.distributed.device_mesh import DeviceMesh
# Initialize device mesh as (2, 4) to represent the topology
# of cross-host(dim 0), and within-host (dim 1).
mesh = DeviceMesh(device_type="cuda", mesh=[[0],[1]])
print(mesh)
