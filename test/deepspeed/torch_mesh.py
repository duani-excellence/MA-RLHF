from torch.distributed.device_mesh import DeviceMesh
# Initialize device mesh as (2, 4) to represent the topology
# of cross-host(dim 0), and within-host (dim 1).
mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
