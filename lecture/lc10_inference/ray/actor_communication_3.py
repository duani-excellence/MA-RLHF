# by xiaodongguaAIGC
# 此代码修改异步收发, 4 种组合, 查看效果
# `def send_to(...` -> `async def send_to(...`
# `def receive(...` -> `async def receive(...`

import ray
import time

ray.init()

@ray.remote
class SenderActor:
    def __init__(self, ):
        self.datas = [1, 2, 3, 4, 5]
    
    async def send_to(self, receiver_actor):
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
    
    async def receive(self, data):
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self,):
        return self.datas
        

send_actor = SenderActor.remote()
rec_actor = RecActor.remote()

futures = ray.get(send_actor.send_to.remote(rec_actor))

# 发送出去的顺序
result = ray.get(futures)
print('receive actor len:', result)

# 接收的顺序
result = ray.get(rec_actor.get_datas.remote())
print('receive actor datas:', result)

