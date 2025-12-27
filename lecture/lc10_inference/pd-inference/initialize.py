
from abc import ABC
import torch.distributed as dist
import torch.multiprocessing as mp

class DistributedStrategy(ABC):
    def __init__(
        self,
        # rank,
        # master_addr,
        # master_port,
        # world_size,
        group_name: str = 'xdg_group',
    ) -> None:
        
        # self.master_addr = master_addr
        # self.master_port = master_port
        # self.rank = rank
        # self.world_size = world_size
        self.group_name = group_name
        
        # self.init_dist(master_addr = master_addr,
        #           master_port = master_port,
        #           rank = rank,
        #           world_size = world_size)
    
def init_dist(
    rank,
    master_addr,
    master_port,
    world_size,
    # group_name,
    ):
    
    # self.master_addr = master_addr
    # self.master_port = master_port
    # self.rank = rank
    # self.world_size = world_size
    # strategy  = DistributedStrategy(group_name)
    
    dist.init_process_group(backend = 'gloo', 
                        init_method = 'tcp://'+ master_addr + ':' + master_port,
                        rank=rank, 
                        world_size=world_size)
    
    print('[group:_]',dist.get_rank(), '/', dist.get_world_size(), 'is initialized')
    
    # dist.barrier()
    # dist.destroy_process_group()
    


if __name__ == '__main__':
    # strategy=DistributedStrategy(
    #     group_name='my_group'
    # )
    mp.spawn(init_dist, args=("127.0.0.1", "12801", 4,), nprocs=4)