import ray
import asyncio
from collections import deque

ray.init()

@ray.remote
class MessageBroker:
    """消息代理，负责消息路由"""
    def __init__(self):
        self.subscribers = {}
        
    def subscribe(self, topic: str, actor_handle):
        """订阅主题"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(actor_handle)
        
    async def publish(self, topic: str, message):
        """发布消息给所有订阅者"""
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                # 异步通知订阅者
                subscriber.on_message.remote(message)
        return len(self.subscribers.get(topic, []))

@ray.remote
class StepWorker:
    """执行步骤的 Worker"""
    def __init__(self, broker_handle):
        self.broker = broker_handle
        self.message_queue = deque()
        self.processing = True
        
    async def on_message(self, message):
        """接收消息回调"""
        self.message_queue.append(message)
        print(f"StepWorker: 收到消息 {message}")
        
    async def run(self):
        """主循环：执行步骤并处理消息"""
        step = 0
        while self.processing:
            step += 1
            print(f"Step {step}: 执行步骤")
            
            # 处理队列中的消息
            while self.message_queue:
                msg = self.message_queue.popleft()
                print(f"Step {step}: 处理消息 {msg}")
            
            await asyncio.sleep(1)
            
        return f"执行了 {step} 步"

@ray.remote
class DataProducer:
    """数据生产者"""
    def __init__(self, broker_handle, topic):
        self.broker = broker_handle
        self.topic = topic
        
    async def produce(self, interval=0.5, count=20):
        """生产数据"""
        for i in range(count):
            message = f"data_{i}"
            await self.broker.publish.remote(self.topic, message)
            print(f"DataProducer: 发布了 {message}")
            await asyncio.sleep(interval)
        return "生产完成"

# 创建组件
broker = MessageBroker.remote()
worker = StepWorker.remote(broker)
producer = DataProducer.remote(broker, "data_topic")

# 订阅主题
ray.get(broker.subscribe.remote("data_topic", worker))

# 并行执行
async def main():
    # 启动 Worker
    worker_task = worker.run.remote()
    
    # 启动生产者
    producer_task = producer.produce.remote(interval=0.3, count=15)
    
    # 等待生产者完成
    producer_result = await producer_task
    print(f"生产者: {producer_result}")
    
    # 等待 5 秒后停止 Worker
    await asyncio.sleep(5)
    
    # 注意：这里需要实现停止逻辑（示例中简化了）
    
    return await worker_task

result = ray.get(main())
print(f"最终结果: {result}")