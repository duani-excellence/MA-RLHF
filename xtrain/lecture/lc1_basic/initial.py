'''
1. 初始化包含两种方法 `init_process_group` 和 `init_device_mesh`, 后者调用前者
    前者可以看成是构建了1-d的设备组, 后者更智能可以定义2-d的设备组, 通常用于多卡GPU初始化
2. `torch.distribute` 包提供创建, 删除, 查看rank号等基础分布式操作
3. 该程序我们仅在单个卡上进行创建, 后续创建多卡的分布式环境
'''
# MASTER_ADDR=127.0.0.1 MASTER_PORT=12801  RANK=0  WORLD_SIZE=1 python initial.py 

import torch
import torch.distributed as dist

# BACKEND = '''gloo''' # gloo, nccl, mpi

def main():
    '''
    pre-process basic initial 
        1. we check enviroment whether avaliable
        2. initital 
        3. check is_initialed
        4. destroy
    '''
    print('-'*100)
    print('torch distributed is avaliable:',
        dist.is_available())

    print('mpi avaliable:', dist.is_mpi_available())
    print('gloo avaliable:', dist.is_gloo_available())
    print('nccl avaliable:', dist.is_nccl_available())

    print('torch distributed is is_initialized:', 
        dist.is_initialized())

    dist.init_process_group(backend="gloo", # gloo, nccl, mpi
                            init_method='env://', # default
                            world_size=1,
                            rank=0,
                            group_name='basic_distributed_env',
                            )

    print('torch distributed is is_initialized:', 
        dist.is_initialized())

    dist.destroy_process_group()
    print('torch distributed is is_initialized:', 
        dist.is_initialized())


def init_tcp():
    '''
    initial with tcp
    '''
    print('-'*100)
    print('init with tcp')
    print('torch distributed is is_initialized:', 
        dist.is_initialized())
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://127.0.0.1:12801',
                            rank=0, 
                            world_size=1)
    print('torch distributed is is_initialized:', 
        dist.is_initialized())
    dist.destroy_process_group()


def init_device_mesh():
    '''
    initial with device mesh
    '''
    print('-'*100)
    print('init with device mesh')
    print('1d-mesh')
    print('torch distributed is is_initialized:', 
        dist.is_initialized())
    dist.init_device_mesh(device_type = 'cpu', 
                            mesh_shape=(1,1))
    print('torch distributed is is_initialized:', 
        dist.is_initialized())
    dist.destroy_process_group()

    print('2d-mesh')
    dist.init_device_mesh(device_type = 'cpu', 
                            mesh_shape=(1,1),
                            mesh_dim_names=("dp", "tp"))
    print('torch distributed is is_initialized:', 
        dist.is_initialized())
    dist.destroy_process_group()

    print('2d-mesh')
    dist.device_mesh.DeviceMesh(device_type = 'cpu', 
                            mesh=[[0]], # [[0,1],[2,3]]
                            # mesh_dim_names=("dp", "tp")
                            )
    print('torch distributed is is_initialized:', 
        dist.is_initialized())
    dist.destroy_process_group()

def init_file():
    # ...
    return


def get_env():
    '''
    check dist detail
    '''
    print('-'*100)
    print('check device mesh env')

    # dist.type
    # dist.init_device_mesh(device_type = 'cpu', 
    #                     mesh_shape=(1,1),
    #                     mesh_dim_names=("dp", "tp"))
    dist.init_process_group(backend="gloo", 
                        init_method='env://', 
                        world_size=1,
                        rank=0,
                        group_name='basic_distributed_env',
                        )
    print('torch distributed is is_initialized:', 
        dist.is_initialized())
    print(dist.get_backend())
    print(dist.get_rank())
    print(dist.get_world_size())
    dist.destroy_process_group()

def group():
    '''
    group management device
    '''
    print('-'*100)
    print('group management device')
    dist.init_process_group(backend = 'gloo', 
                        init_method = 'tcp://127.0.0.1:12803',
                        rank=0, 
                        world_size=1)
    print('torch distributed is is_initialized:', 
            dist.is_initialized())

    group = torch.distributed.new_group(ranks =[0],
                                backend='gloo')
    print(group)
    print(dist.get_group_rank(group = group,
                              global_rank=0 ))
    print(torch.distributed.get_process_group_ranks(group))
    dist.destroy_process_group()



if __name__ == "__main__":
    main()
    init_tcp()
    get_env()
    init_device_mesh()
    group()
    print('exit')

