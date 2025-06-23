# python store.py 


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import time

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    store 存储共享变量
    '''
    if rank == 0:
        print('-'*100)
        print('store')
    
    # dist.init_process_group(backend = 'gloo', 
    #                         init_method = 'tcp://127.0.0.1:' + master_port,
    #                         rank=rank, 
    #                         world_size=world_size)
    
    store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))

    # add & get
    store.add("first_key", 1)
    store.add("first_key", 6)
    print(store.get("first_key"))

    # append
    store.append("second_key", "xiao")
    store.append("second_key", "dong")
    store.append("second_key", "gua")
    print(store.get("second_key"))

    # check
    store.add("third_key", 0)
    print(store.check(["third_key"]))
    print(store.check(["null"])) # 不存在就返回False

    # compare_set
    store.set("key", "first_value")
    store.compare_set("key", 
                      "first_value", 
                      "second_value")
    print(store.get("key"))

    # delete
    print(store.delete_key("key"))
    print(store.delete_key("empty_key"))

    # multi_get
    store.set("first_key", "po")
    store.set("second_key", "tato")
    print(store.multi_get(["first_key", "second_key"]))

    # multi_set
    store.multi_set(["first_key", "second_key"], ["po", "tato"])
    print(store.get("first_key"))

    # num
    print(store.num_keys())

    # wait
    # store.set_timeout(timedelta(seconds=3))
    # if rank == 0:
    #     time.sleep(1)
    #     store.set("bad_key", "haha")
    # else:
    #     store.wait(["bad_key"])
    #     print('test')
    
    # dist.destroy_process_group()



def tcp_store(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    tcp store 存储共享变量
    '''
    if rank == 0:
        print('-'*100)
        print('tcp store')

    if rank == 0:
        server_store = dist.TCPStore("127.0.0.1", 12884, 2, True, timedelta(seconds=30))
        server_store.set("first_key", "first_value")
    else:
        client_store = dist.TCPStore("127.0.0.1", 12884, 2, False)
        client_store.wait(["first_key"])
        print(client_store.get("first_key"))




if __name__ == '__main__':

    # 采用 torch 自带的多线程库来模拟6个进程执行
    mp.spawn(run, args=("127.0.0.1", "12801", 3, ), nprocs=3)
    mp.spawn(tcp_store, args=("127.0.0.1", "12801", 3, ), nprocs=3)

    # HashStore, FileStore, PrefixStore
