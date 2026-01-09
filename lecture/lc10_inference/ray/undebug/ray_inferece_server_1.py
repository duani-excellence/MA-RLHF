import ray
import asyncio
from typing import Optional
import time

# 初始化 Ray
ray.init()

class DataSender:
    """发送数据的 Actor"""
    def __init__(self, receiver_handle):
        self.receiver_handle = receiver_handle
        
    async def send_data(self, data):
        """异步发送数据给接收者"""
        # 这里使用异步调用，不阻塞发送者
        await self.receiver_handle.receive_data.remote(data)
        print(f"DataSender: 已发送数据 {data}")
        return True
    
    async def continuous_send(self):
        """持续发送数据"""
        for i in range(10):
            await self.send_data(f"data_{i}")
            await asyncio.sleep(1)  # 模拟异步发送间隔
        return "发送完成"

class StepExecutor:
    """执行步骤并监听数据的 Actor"""
    def __init__(self):
        self.data_queue = []
        self.running = True
        self.step_count = 0
        
    async def receive_data(self, data):
        """异步接收数据（这个方法会被其他 Actor 调用）"""
        self.data_queue.append(data)
        print(f"StepExecutor: 收到数据 {data}，当前队列长度 {len(self.data_queue)}")
        
    async def process_data(self):
        """处理队列中的数据"""
        while self.data_queue:
            data = self.data_queue.pop(0)
            print(f"StepExecutor: 处理数据 {data}")
            # 这里可以添加数据处理逻辑
            await asyncio.sleep(0.1)  # 模拟处理时间
            
    async def step(self):
        """执行一步操作"""
        self.step_count += 1
        print(f"StepExecutor: 执行第 {self.step_count} 步")
        
        # 处理接收到的数据
        await self.process_data()
        
        # 模拟步骤执行时间
        await asyncio.sleep(0.5)
        return self.step_count
    
    async def continuous_step(self):
        """持续执行步骤"""
        while self.running:
            await self.step()
        return "执行完成"
    
    async def stop(self):
        """停止执行"""
        self.running = False

# 创建 Actor
step_executor = StepExecutor.remote()
data_sender = DataSender.remote(step_executor)

# 并行执行
async def main():
    # 启动 StepExecutor 的持续执行
    step_task = step_executor.continuous_step.remote()
    
    # 启动 DataSender 的持续发送
    send_task = data_sender.continuous_send.remote()
    
    # 等待一段时间后停止
    await asyncio.sleep(12)
    
    # 停止 StepExecutor
    await step_executor.stop.remote()
    
    # 获取结果
    step_result = await step_task
    send_result = await send_task
    
    return step_result, send_result

# 运行
results = ray.get(main())
print(f"结果: {results}")