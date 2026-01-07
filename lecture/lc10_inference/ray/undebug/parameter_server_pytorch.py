import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import time

# 初始化Ray
ray.init()

# 定义简单的MLP分类模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 参数服务器Actor
@ray.remote
class ParameterServer:
    def __init__(self):
        # 初始化模型
        self.model = SimpleMLP()
        self.version = 0
        # Optimizer 实现在一个独立节点, 其他 worker GPU 可以不维护优化器参数
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    
    def get_weights(self):
        """获取当前模型权重"""
        return {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}, self.version
    
    def update_weights(self, gradients: Dict[str, torch.Tensor], worker_id: str):
        """更新模型权重"""
        print(f"参数服务器: 收到来自工作节点 {worker_id} 的梯度")
        
        # 应用梯度
        for name, param in self.model.named_parameters():
            if name in gradients:
                """
                TODO: 梯度需要 reduce
                """
                param.grad = gradients[name].to(param.device)
        
        # 执行优化步骤
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.version += 1
        print(f"参数服务器: 权重已更新到版本 {self.version}")
        return self.version

# 工作节点Actor
@ray.remote
class Worker:
    def __init__(self, worker_id: str, parameter_server):
        self.worker_id = worker_id
        self.parameter_server = parameter_server
        
        # 本地模型副本
        self.model = SimpleMLP()
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, batch_size=32):
        """执行一个训练步骤"""
        print(f"工作节点 {self.worker_id}: 开始训练步骤")
        
        # 1. 从参数服务器获取最新权重
        weights, version = ray.get(self.parameter_server.get_weights.remote())
        self.model.load_state_dict(weights)
        print(f"工作节点 {self.worker_id}: 获取到版本 {version} 的权重")
        
        # 2. 生成随机训练数据
        inputs = torch.randn(batch_size, 10)  # 10维输入
        labels = torch.randint(0, 2, (batch_size,))  # 二分类标签
        
        # 3. 前向传播
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        # 4. 反向传播计算梯度
        self.model.zero_grad()
        loss.backward()
        
        # 5. 提取梯度
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().clone()
        
        print(f"工作节点 {self.worker_id}: 计算完成，损失={loss.item():.4f}")
        
        # 6. 将梯度发送到参数服务器
        new_version = ray.get(
            self.parameter_server.update_weights.remote(gradients, self.worker_id)
        )
        
        return loss.item(), new_version

def main():
    print("=== PyTorch MLP 参数服务器梯度更新 ===")
    
    # 创建参数服务器
    ps = ParameterServer.remote()
    
    # 创建多个工作节点
    workers = [Worker.remote(f"Worker-{i}", ps) for i in range(3)]
    
    # 模拟分布式训练
    print("\n=== 开始分布式训练 ===")
    for epoch in range(3):
        print(f"\n--- 第 {epoch+1} 轮训练 ---")
        
        # 所有工作节点并行训练
        futures = []
        for i, worker in enumerate(workers):
            future = worker.train_step.remote(batch_size=16)
            futures.append(future)
        
        # 收集结果
        results = ray.get(futures)
        
        for i, (loss, version) in enumerate(results):
            print(f"工作节点 {i}: 损失={loss:.4f}, 更新到版本={version}")
        
        time.sleep(0.5)  # 模拟训练间隔
    
    print("\n=== 训练完成 ===")
    
    # 获取最终模型权重
    final_weights, final_version = ray.get(ps.get_weights.remote())
    print(f"最终模型版本: {final_version}")
    print(f"权重参数数量: {len(final_weights)}")
    for name, weight in final_weights.items():
        print(f"  {name}: {weight.shape}")

if __name__ == "__main__":
    main()