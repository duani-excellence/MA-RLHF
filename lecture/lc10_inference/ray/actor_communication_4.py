# by xiaodongguaAIGC

import ray
import time

ray.init()

@ray.remote
class SenderActor:
    def __init__(self, datas):
        self.datas = datas
    
    async def send_to(self, receiver_actor):
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
    # async def receive(self, data):
    def receive(self, data):
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self,):
        return self.datas
        
send_actor_1 = SenderActor.remote(list(range(1,5)))
send_actor_2 = SenderActor.remote(list(range(15,20)))
rec_actor = RecActor.remote()

print('---2actor 异步发送---')
futures_1 = ray.get(send_actor_1.send_to.remote(rec_actor))
futures_2 = ray.get(send_actor_2.send_to.remote(rec_actor))

result = ray.get(futures_2+futures_1)  # 有顺序性
print('2 send actor to recv actor, datalen:', result)
result = ray.get(futures_1+futures_2)  # 有顺序性
print('2 send actor to recv actor, datalen:', result)

# 接收的顺序
result = ray.get(rec_actor.get_datas.remote())
print('receive actor datas:', result)
