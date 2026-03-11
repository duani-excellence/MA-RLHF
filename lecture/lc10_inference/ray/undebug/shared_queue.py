import ray
import time
from collections import deque

ray.init()

# 共享队列Actor
@ray.remote
class SharedQueue:
    def __init__(self, max_size=10):
        self.queue = deque(maxlen=max_size)
        self.processed_count = 0
    
    def push(self, item):
        if len(self.queue) < self.queue.maxlen:
            self.queue.append(item)
            return True
        return False
    
    def pop(self):
        if self.queue:
            self.processed_count += 1
            return self.queue.popleft()
        return None
    
    def size(self):
        return len(self.queue)
    
    def get_stats(self):
        return {
            "current_size": len(self.queue),
            "processed_total": self.processed_count
        }

# 生产者Actor
@ray.remote
class Producer:
    def __init__(self, queue, producer_id):
        self.queue = queue
        self.id = producer_id
    
    def produce(self, items):
        for item in items:
            while True:
                success = ray.get(self.queue.push.remote(item))
                if success:
                    print(f"生产者{self.id} 生产: {item}")
                    break
                time.sleep(0.1)  # 队列满时等待
            time.sleep(0.5)

# 消费者Actor
@ray.remote
class Consumer:
    def __init__(self, queue, consumer_id):
        self.queue = queue
        self.id = consumer_id
    
    def consume(self):
        while True:
            item = ray.get(self.queue.pop.remote())
            if item is not None:
                print(f"消费者{self.id} 消费: {item}")
                # 模拟处理时间
                time.sleep(1)
            else:
                time.sleep(0.5)

# 创建共享队列
queue = SharedQueue.remote(max_size=5)

# 创建生产者和消费者
producer1 = Producer.remote(queue, "P1")
producer2 = Producer.remote(queue, "P2")
consumer1 = Consumer.remote(queue, "C1")

# 启动流式处理（在实际使用中，这些应该在不同的进程中）
import threading

def run_producer1():
    ray.get(producer1.produce.remote([f"任务{i}" for i in range(1, 6)]))

def run_producer2():
    ray.get(producer2.produce.remote([f"作业{i}" for i in range(1, 6)]))

def run_consumer():
    ray.get(consumer1.consume.remote())

# 启动线程模拟并发
# 注意：实际Ray应用中，这些会在不同节点上运行
print("开始流式处理...")
t1 = threading.Thread(target=run_producer1)
t2 = threading.Thread(target=run_producer2)
t3 = threading.Thread(target=run_consumer)

t1.start()
t2.start()
time.sleep(1)
t3.start()

t1.join()
t2.join()
# 消费者会一直运行，这里我们等待一会儿
time.sleep(10)
print("流式处理演示结束")