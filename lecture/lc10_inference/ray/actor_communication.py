import ray
import time

ray.init()

# Actor类
@ray.remote
class SimpleActor:
    def __init__(self, name):
        self.name = name
    
    def send_to(self, target_actor, message):
        # 直接调用目标Actor的方法
        return target_actor.receive.remote(f"来自{self.name}: {message}")
    
    def receive(self, message):
        print(f"{self.name} 收到消息: {message}")
        return "OK"

# 创建两个Actor
actor_user = SimpleActor.remote("Actor User")
actor_server = SimpleActor.remote("Actor Server")

future = ray.get(actor_user.send_to.remote(actor_server, "hello, 1+1=?"))
message = ray.get(future)  # 确保消息被处理

future = ray.get(actor_server.send_to.remote(actor_user, "1+1=2"))
ray.get(future)

time.sleep(1)

ray.shutdown()
