# by xiaodongguaAIGC
# 创建一个共享队列
# Sender 随机延时发送 10 个数据, 至队列
# receiver 监听共享队列，将数据放置在内部列表中

import ray
import time
from collections import deque
import random
# import 

ray.init()

@ray.remote
class SharedQueue:
    def __init__(self, max_size=100):
        self.queue = deque(maxlen=max_size)
        self.processed_count = 0
    
    def push(self, item):
        if len(self.queue) < self.queue.maxlen:
            self.queue.append(item)
            # print(len(self.queue))
            return True
        return False
    
    def pop(self):
        if self.queue:
            self.processed_count += 1
            return self.queue.popleft()
        return None
    
    def get_stats(self):
        return {
            "queue buffer size:": len(self.queue),
            
        }

@ray.remote
class Actor:
    def __init__(self, buffer, name):
        self.datas = [i for i in range(10)]
        self.queue = buffer
        self.tasks = []
        self.name = name
    
    async def send_to(self):
        futures = []
        for i in self.datas:
            
            future = self.queue.push.remote(i)
            futures.append(future)
            # sleep
            time.sleep(random.random()/2.0)
        return futures
    
    async def receive(self):
        while 1:
            info = ray.get(self.queue.get_stats.remote())
            print(info)
            
            idx = ray.get(self.queue.pop.remote())
            if idx != None:
                self.tasks.append(idx)
                print('receiver get from queue:', idx)
                print(self.tasks)
            
            if len(self.tasks) == 10:
                break
            
            time.sleep(random.random())
        return len(self.datas)
    
    def get_datas(self,):
        return self.datas


queue = SharedQueue.remote(max_size=5)

# 创建生产者和消费者
sender = Actor.remote(queue, "sender")
receiver = Actor.remote(queue, "receiver")

futures_recv = receiver.receive.remote()
futures_send = sender.send_to.remote()
ray.get(futures_recv)
ray.get(futures_send)
time.sleep(5)





