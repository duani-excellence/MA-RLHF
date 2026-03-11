import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# 初始化Ray
ray.init()

# ========== 第一部分：参数服务器梯度更新 ==========
print("=" * 50)
print("第一部分：参数服务器梯度更新 (PyTorch MLP)")
print("=" * 50)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

@ray.remote
class ParamServer:
    def __init__(self):
        self.model = SimpleMLP()
        self.version = 0
    
    def get_params(self):
        return {k: v.clone() for k, v in self.model.state_dict().items()}, self.version
    
    def apply_gradients(self, gradients):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param -= 0.01 * gradients[name]
        self.version += 1
        return self.version

@ray.remote
class WorkerNode:
    def __init__(self, worker_id, server):
        self.id = worker_id
        self.server = server
        self.model = SimpleMLP()
    
    def train_step(self):
        # 获取参数
        params, _ = ray.get(self.server.get_params.remote())
        self.model.load_state_dict(params)
        
        # 生成数据
        data = torch.randn(16, 10)
        target = torch.randint(0, 2, (16,))
        
        # 计算梯度
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # 提取梯度
        grads = {name: param.grad.clone() for name, param in self.model.named_parameters()}
        
        # 更新服务器
        new_ver = ray.get(self.server.apply_gradients.remote(grads))
        return f"Worker {self.id}: loss={loss.item():.3f}, ver={new_ver}"

# 运行参数服务器示例
print("\n运行参数服务器梯度更新...")
server = ParamServer.remote()
workers = [WorkerNode.remote(f"W{i}", server) for i in range(2)]

futures = [w.train_step.remote() for w in workers]
results = ray.get(futures)
for r in results:
    print(r)

# ========== 第二部分：多节点训练 ==========
print("\n" + "=" * 50)
print("第二部分：多节点训练 (ActorA + ActorB)")
print("=" * 50)

@ray.remote
class ActorA:
    def __init__(self, actor_b_ref):
        self.actor_b = actor_b_ref
    
    def generate_and_send(self):
        # 生成随机数据和标签
        x = np.random.randint(0, 50, 5).tolist()  # 5个随机数
        y = np.random.randint(0, 2)  # 0或1
        
        # 简单预测规则
        y_pred = 1 if sum(x) > 100 else 0
        
        # 发送给ActorB
        result = ray.get(self.actor_b.process.remote(x, y, y_pred))
        return f"ActorA: 发送数据 x={x}, y={y}, y_pred={y_pred} | {result}"

@ray.remote
class ActorB:
    def __init__(self):
        # 简单的embedding层
        self.embedding = nn.Embedding(50, 8)
        self.fc = nn.Linear(8, 2)
        self.losses = []
    
    def process(self, x, y_true, y_pred_a):
        # 处理输入
        x_tensor = torch.LongTensor(x)
        embedded = self.embedding(x_tensor)
        pooled = torch.mean(embedded, dim=0)
        output = self.fc(pooled)
        
        # 计算损失
        target = torch.LongTensor([y_true])
        loss = F.cross_entropy(output.unsqueeze(0), target)
        self.losses.append(loss.item())
        
        # 预测
        pred = torch.argmax(output).item()
        
        return (f"ActorB: 收到数据 | 预测={pred}, 真实={y_true}, "
                f"ActorA预测={y_pred_a}, 损失={loss.item():.3f}, "
                f"平均损失={np.mean(self.losses):.3f}")

# 运行多节点训练示例
print("\n运行多节点训练...")
actor_b = ActorB.remote()
actor_a = ActorA.remote(actor_b)

# 模拟流式处理
for i in range(5):
    result = ray.get(actor_a.generate_and_send.remote())
    print(result)
    time.sleep(0.3)

print("\n" + "=" * 50)
print("两个功能都已完成!")
print("=" * 50)

# 关闭Ray
ray.shutdown()