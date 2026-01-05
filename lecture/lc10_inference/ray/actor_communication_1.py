# by xiaodongguaAIGC

import ray
import time

ray.init()

@ray.remote
class SenderActor:
    def __init__(self, ):
        self.datas = [12, 32, 4, 8 , 10]
    
    def send_to(self, receiver_actor, data):
        return receiver_actor.receive.remote(data)
    
    def get_datas(self,):
        return self.datas

@ray.remote      
class RecActor:
    def __init__(self, ):
        self.datas = []
    
    def receive(self, data):
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self,):
        return self.datas
        

send_actor = SenderActor.remote()
rec_actor = RecActor.remote()

datas = ray.get(send_actor.get_datas.remote())
print(datas)

for i in datas:
    future = ray.get(send_actor.send_to.remote(rec_actor, i))
    # print(future)
    
    result = ray.get(future)
    print('receive actor len:', result)
    
    result = ray.get(rec_actor.get_datas.remote())
    print('receive actor datas:', result)
    # rec_actor.print()