import ray
from ray import serve
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

ray.init()
serve.start()

class DataSender:
    """发送者服务"""
    def __init__(self):
        self.step_executor_handle: Optional[ray.actor.ActorHandle] = None
        
    def set_receiver(self, receiver_handle):
        """设置接收者"""
        self.step_executor_handle = receiver_handle
        
    async def send(self, data: str) -> bool:
        """发送数据（异步）"""
        if self.step_executor_handle:
            # 使用 fire-and-forget 模式，不等待响应
            self.step_executor_handle.receive_data.remote(data)
            return True
        return False
    
    async def stream_send(self, data_stream: List[str]):
        """流式发送数据"""
        for data in data_stream:
            await self.send(data)
            await asyncio.sleep(0.5)

class StepExecutor:
    """执行步骤的服务"""
    def __init__(self):
        self.data_buffer = []
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._start_step_loop()
        
    def _start_step_loop(self):
        """启动步骤循环的后台线程"""
        def step_loop():
            step_count = 0
            while True:
                # 执行步骤
                step_count += 1
                print(f"Step {step_count}: 执行操作")
                
                # 处理缓冲的数据
                with self.lock:
                    if self.data_buffer:
                        data = self.data_buffer.pop(0)
                        print(f"Step {step_count}: 处理数据 {data}")
                
                time.sleep(1)  # 步骤间隔
                
        thread = threading.Thread(target=step_loop, daemon=True)
        thread.start()
        
    def receive_data(self, data: str):
        """接收数据（线程安全）"""
        with self.lock:
            self.data_buffer.append(data)
            print(f"收到数据: {data}, 缓冲区大小: {len(self.data_buffer)}")

# 部署服务
StepExecutor.deploy()
DataSender.deploy()

# 获取服务句柄
step_executor = StepExecutor.get_handle()
data_sender = DataSender.get_handle()

# 设置接收者
ray.get(data_sender.set_receiver.remote(step_executor))

# 发送数据流
data_stream = [f"payload_{i}" for i in range(20)]
ray.get(data_sender.stream_send.remote(data_stream))