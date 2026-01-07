import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import time

# 初始化Ray
ray.init()

# 定义ActorA使用的简单模型（用于生成预测）
class ActorAModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=10):
        super(ActorAModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # 二分类输出
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 输出概率
        return x

# 定义ActorB使用的模型（包含Embedding层）
class ActorBModel(nn.Module):
    def __init__(self, vocab_size=100, embedding_dim=16, hidden_size=32):
        super(ActorBModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # 二分类输出
    
    def forward(self, x):
        # x是整数列表，需要转换为LongTensor
        x = torch.LongTensor(x)
        x = self.embedding(x)
        x = torch.mean(x, dim=0)  # 对序列取平均
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# ActorA：生成数据并进行初始预测
@ray.remote
class ActorA:
    def __init__(self, actor_b_ref):
        self.model = ActorAModel()
        self.actor_b = actor_b_ref
        self.data_count = 0
    
    def generate_data_batch(self, batch_size=8, seq_length=5):
        """生成一批数据"""
        batch_data = []
        
        for _ in range(batch_size):
            # 生成随机数字列表（模拟输入序列）
            x = np.random.randint(0, 100, size=seq_length).tolist()
            
            # 生成二分类标签
            y = np.random.randint(0, 2)
            
            # 使用本地模型生成预测
            x_tensor = torch.FloatTensor(x)
            y_pred_prob = self.model(x_tensor)
            y_pred = 1 if y_pred_prob.item() > 0.5 else 0
            
            batch_data.append((x, y, y_pred, y_pred_prob.item()))
        
        self.data_count += len(batch_data)
        return batch_data
    
    def stream_data_to_actor_b(self, num_batches=5):
        """流式传输数据到ActorB"""
        print(f"ActorA: 开始向ActorB流式传输数据 ({num_batches}批次)")
        
        for batch_idx in range(num_batches):
            # 生成一批数据
            batch_data = self.generate_data_batch(batch_size=4)
            
            print(f"\nActorA: 生成批次 {batch_idx+1}, 包含 {len(batch_data)} 个样本")
            
            # 将数据传输给ActorB进行训练
            result = ray.get(self.actor_b.train_on_batch.remote(batch_data, batch_idx))
            
            print(f"ActorA: ActorB训练结果 - {result}")
            
            # 模拟流式处理延迟
            time.sleep(0.5)
        
        return f"传输完成，共生成 {self.data_count} 个样本"

# ActorB：接收数据并进行训练
@ray.remote
class ActorB:
    def __init__(self):
        self.model = ActorBModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.train_count = 0
        self.total_loss = 0.0
    
    def train_on_batch(self, batch_data: List[Tuple], batch_idx: int):
        """使用一批数据进行训练"""
        print(f"\nActorB: 收到批次 {batch_idx+1}, 开始训练")
        
        batch_loss = 0.0
        correct = 0
        total = len(batch_data)
        
        for x, y_true, y_pred_a, _ in batch_data:
            # 将数据转换为模型输入
            self.optimizer.zero_grad()
            
            # 前向传播
            output_prob = self.model(x)
            output = output_prob.item()
            
            # 计算损失（使用ActorA的真实标签）
            target = torch.FloatTensor([float(y_true)])
            loss = self.criterion(output_prob, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            batch_loss += loss.item()
            
            # 统计ActorA预测的准确率
            predicted_b = 1 if output > 0.5 else 0
            if predicted_b == y_true:
                correct += 1
            
            self.train_count += 1
        
        avg_loss = batch_loss / total
        self.total_loss += batch_loss
        
        accuracy = correct / total * 100
        
        # 显示样本示例
        if batch_idx == 0:
            print(f"ActorB: 样本示例 - x={batch_data[0][0]}, y_true={batch_data[0][1]}, "
                  f"y_pred_a={batch_data[0][2]}, y_pred_b={1 if self.model(batch_data[0][0]).item() > 0.5 else 0}")
        
        return {
            "batch": batch_idx + 1,
            "avg_loss": round(avg_loss, 4),
            "accuracy": round(accuracy, 2),
            "total_trained": self.train_count,
            "total_avg_loss": round(self.total_loss / self.train_count, 4)
        }
    
    def evaluate(self, test_data):
        """评估模型性能"""
        correct = 0
        total = len(test_data)
        
        for x, y_true, _, _ in test_data:
            output_prob = self.model(x)
            predicted = 1 if output_prob.item() > 0.5 else 0
            
            if predicted == y_true:
                correct += 1
        
        accuracy = correct / total * 100
        return {
            "test_samples": total,
            "accuracy": round(accuracy, 2)
        }
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        embedding_params = sum(p.numel() for p in self.model.embedding.parameters())
        return {
            "total_parameters": total_params,
            "embedding_parameters": embedding_params,
            "layers": [str(module) for module in self.model.children()]
        }

def main():
    print("=== Ray 多节点 LLM-RL 训练任务部署 ===")
    print("ActorA: 生成数据并初步预测")
    print("ActorB: 包含Embedding层的模型训练\n")
    
    # 创建ActorB
    actor_b = ActorB.remote()
    
    # 创建ActorA，传入ActorB的引用
    actor_a = ActorA.remote(actor_b)
    
    # 1. 先获取ActorB模型信息
    model_info = ray.get(actor_b.get_model_info.remote())
    print(f"ActorB模型信息:")
    print(f"  总参数: {model_info['total_parameters']}")
    print(f"  Embedding参数: {model_info['embedding_parameters']}")
    print(f"  层结构: {model_info['layers'][0]}")
    
    # 2. ActorA向ActorB流式传输数据进行训练
    print("\n=== 开始流式训练 ===")
    stream_result = ray.get(actor_a.stream_data_to_actor_b.remote(num_batches=8))
    print(f"\nActorA: {stream_result}")
    
    # 3. 生成测试数据评估ActorB模型
    print("\n=== 生成测试数据评估模型 ===")
    
    # 生成测试数据（使用ActorA的数据生成方法）
    test_data = ray.get(actor_a.generate_data_batch.remote(batch_size=10))
    
    # 评估ActorB模型
    eval_result = ray.get(actor_b.evaluate.remote(test_data))
    print(f"ActorB评估结果:")
    print(f"  测试样本数: {eval_result['test_samples']}")
    print(f"  准确率: {eval_result['accuracy']}%")
    
    # 4. 演示实时交互
    print("\n=== 实时交互演示 ===")
    
    # 生成新的实时数据
    new_data = ray.get(actor_a.generate_data_batch.remote(batch_size=3))
    
    for i, (x, y_true, y_pred_a, prob_a) in enumerate(new_data):
        # ActorB进行预测
        prob_b = ray.get(actor_b.model.forward.remote(x))
        
        print(f"样本 {i+1}:")
        print(f"  输入: {x}")
        print(f"  真实标签: {y_true}")
        print(f"  ActorA预测: {y_pred_a} (概率: {prob_a:.3f})")
        print(f"  ActorB预测: {1 if prob_b > 0.5 else 0} (概率: {prob_b:.3f})")
        print(f"  是否一致: {'是' if y_pred_a == (1 if prob_b > 0.5 else 0) else '否'}")
    
    print("\n=== 训练部署完成 ===")

if __name__ == "__main__":
    main()