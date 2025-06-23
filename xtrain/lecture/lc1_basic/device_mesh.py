# python group.py 
# MASTER_ADDR=127.0.0.1 MASTER_PORT=12801  RANK=0  WORLD_SIZE=6 python device_mesh.py 
# todo:
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def device_map():
    '''
    device map
    创建2d mesh, 组内及组间是如何通信的？
    '''
    print('-'*100)
    print('device map')

    # os.environ['MASTER_PORT'] = master_port


    print(master_port) # 每个设备的端口都不一样？

    dist.device_mesh.init_device_mesh(device_type = 'cpu', 
                        #    mesh=[[0, 2, 4],[1, 3, 5]],
                            mesh_shape=(2,3)
                            )
    print('torch distributed is is_initialized:', 
            dist.is_initialized())
    
    print('current process rank:', dist.get_rank())

    dist.destroy_process_group()


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

if __name__ == '__main__':
    master_port = find_free_port()
    os.environ['MASTER_PORT'] = master_port
    device_map()