import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import time
import random

def init_distributed_gloo():
    """类似 torch.distributed.init_process_group 的初始化"""
    
    ray.init(_node_ip_address="0.0.0.0")
    
    pg = placement_group(
        [{"CPU": 1}] * 4,  # 4个worker，每个1CPU+1GPU
    )
    ray.get(pg.ready())
    
    @ray.remote(
        num_cpus=1,
        # num_gpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True
        )
    )
    class Worker:
        def __init__(self, rank, world_size):
            self.rank = rank
            self.world_size = world_size
            self.stop = False
        
        def set_stop(self,):
            self.stop = True
        
        
        def init_process_group(self, master_addr, master_port):
            import os
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['RANK'] = str(self.rank)
            os.environ['LOCAL_RANK'] = str(self.rank)
            
            # 初始化Gloo后端
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )
            
            print(f"Worker {self.rank}: {dist.get_rank()}/{dist.get_world_size()}")
            A = torch.randn(100)
            dist.all_reduce(A)
            
            return True
    
    # 创建workers
    world_size = 4
    master_addr = ray.util.get_node_ip_address()
    # master_addr = '127.0.0.1'
    master_port = "29500"
    
    print('create worker')
    workers = [Worker.remote(i, world_size) for i in range(world_size)]
    
    results = ray.get([
        w.init_process_group.remote(master_addr, master_port) 
        for w in workers
    ])
    
    time.sleep(5)
    
    print(f"All workers initialized. Sum test: {results}")

if __name__ == "__main__":
    init_distributed_gloo()