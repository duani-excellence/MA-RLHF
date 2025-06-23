# python p2p_op.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    执行 p2p 批量异步传输, 一次性定义好传输数据和操作, 塞入到`batch_isend_irecv`
    并执行`req.wait()` 完成所有的传输/接收对
    '''
    dist.init_process_group(backend = 'gloo', 
                            init_method = 'tcp://'+ master_addr + ':' + master_port,
                            rank=rank, 
                            world_size=world_size)

    send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
    recv_tensor = torch.zeros(2, dtype=torch.float32)

    # 环形通信
    send_op = dist.P2POp(
        dist.isend, send_tensor, (rank + 1) % world_size)
    print(send_op)
    recv_op = dist.P2POp(
        dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size
    )
    print(recv_op)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    print('Rank ', rank, ' has data ', recv_tensor)
    dist.destroy_process_group()

if __name__ == '__main__':
    # 采用 torch 自带的多线程库来模拟4个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 4, ), nprocs=4)