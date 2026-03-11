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
    
    # 异步接收
    async def receive(self, data):
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self,):
        return self.datas
        
print('---异步接收通信---')
send_actor = SenderActor.remote()
rec_actor = RecActor.remote()

futures = ray.get(send_actor.send_to.remote(rec_actor))
result = ray.get(futures) # 类似 barrier, 阻塞保证接收方完成所有数据接收
print('receive actor len:', result)

# 打印接收的顺序
result = ray.get(rec_actor.get_datas.remote())
print('receive actor datas:', result)

