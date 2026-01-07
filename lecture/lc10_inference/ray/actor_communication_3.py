# by xiaodongguaAIGC

import ray
import time

ray.init()

@ray.remote
class SenderActor:
    def __init__(self, ):
        self.datas = [1, 2, 3, 4, 5]
    
    def send_to(self, receiver_actor):
        # 改写, 批量异步发送, 返回handle, 不管接收方是否收到
        futures = []
        for i in self.datas:
            future = receiver_actor.receive.remote(i)
            futures.append(future)
        return futures
    
    def get_datas(self,):
        return self.datas

@ray.remote      
class RecActor:
    def __init__(self, ):
        self.datas = []
    
    # 同步接收
    def receive(self, data):
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self,):
        return self.datas
        
send_actor = SenderActor.remote()
rec_actor = RecActor.remote()

print('---同步接收通信---')
futures = ray.get(send_actor.send_to.remote(rec_actor))
result = ray.get(futures) # 类似 barrier, 阻塞保证接收方完成所有数据接收
print('receive actor len:', result)

# 接收的顺序
result = ray.get(rec_actor.get_datas.remote())
print('receive actor datas:', result)

# 为什么发送异步, 接收方的数据不会打乱?
# 由于Ray的Actor模型保证同一个Actor的方法调用是顺序执行的，
# 并且发送方是按顺序发送的，所以接收方按顺序接收。即使发送是异步的（即非阻塞发送），
# 但是发送的顺序是确定的，且接收方顺序处理，所以接收顺序与发送顺序一致。
# Actor内部方法执行是串行的：即使有多个并发调用，RecActor也会按顺序一个一个执行receive()方法

