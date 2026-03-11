# trainer_actor.py
"""
训练节点Actor实现
"""

import ray
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Dict, Any

from config import TRAIN_CONFIG
from models import SharedLanguageModel
from data_utils import TrainingSample


@ray.remote(num_gpus=0.5)
class TrainerActor:
    """训练节点Actor：接收数据并训练模型"""
    
    def __init__(self, trainer_id="Trainer-1", config=None):
        if config is None:
            config = TRAIN_CONFIG
        
        self.trainer_id = trainer_id
        self.config = config
        
        # 设置GPU
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        
        # 初始化模型
        self.model = SharedLanguageModel().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config["learning_rate"]
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练队列
        self.training_queue = deque(maxlen=config["queue_size"])
        self.training_step = 0
        self.params_version = 0
        
        # 统计信息
        self.total_samples = 0
        self.loss_history = []
        
        print(f"✅ 训练节点 {trainer_id} 初始化完成，使用 {self.device}")
    
    def receive_generated_data(self, batch_data: List[Dict]) -> bool:
        """接收生成节点发送的数据"""
        if not batch_data:
            return False
        
        # 将数据添加到训练队列
        for data in batch_data:
            # 转换字典到TrainingSample对象
            sample = TrainingSample(
                prompt=data["prompt"],
                generated=data["generated"],
                input_ids=data["input_ids"],
                labels=data["labels"],
                generator_id=data["generator_id"],
                timestamp=data["timestamp"]
            )
            self.training_queue.append(sample)
        
        new_samples = len(batch_data)
        self.total_samples += new_samples
        
        print(f"📥 训练节点 {self.trainer_id}: 收到 {new_samples} 个样本，队列大小: {len(self.training_queue)}")
        return True
    
    def train_step(self, batch_size=None) -> Dict[str, Any]:
        """执行一个训练步骤"""
        if batch_size is None:
            batch_size = self.config["batch_size"]
        
        # 检查队列是否足够
        if len(self.training_queue) < batch_size:
            return {
                "status": "waiting",
                "queue_size": len(self.training_queue),
                "required": batch_size
            }
        
        # 从队列中随机采样一批数据
        indices = np.random.choice(len(self.training_queue), batch_size, replace=False)
        batch_samples = [self.training_queue[i] for i in indices]
        
        # 准备训练数据
        input_tensors = []
        label_tensors = []
        
        for sample in batch_samples:
            input_tensors.append(sample.input_ids)
            label_tensors.append(sample.labels)
        
        # 转换为tensor并移动到GPU
        try:
            # 获取最大长度
            max_len = max(tensor.size(0) for tensor in input_tensors)
            
            # 填充到相同长度
            padded_inputs = []
            padded_labels = []
            
            for inp, lab in zip(input_tensors, label_tensors):
                pad_len = max_len - inp.size(0)
                if pad_len > 0:
                    inp_padded = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
                    lab_padded = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
                else:
                    inp_padded = inp
                    lab_padded = lab
                
                padded_inputs.append(inp_padded)
                padded_labels.append(lab_padded)
            
            # 堆叠tensors
            input_tensor = torch.stack(padded_inputs).to(self.device)
            label_tensor = torch.stack(padded_labels).to(self.device)
        
        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return {"status": "error", "message": str(e)}
        
        # 训练步骤
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            # 前向传播
            logits = self.model(input_tensor)
            
            # 计算损失（忽略标签为-100的位置）
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                label_tensor.view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config["grad_clip"]
            )
            
            # 优化器步骤
            self.optimizer.step()
        
        except Exception as e:
            print(f"❌ 训练步骤失败: {e}")
            return {"status": "error", "message": str(e)}
        
        # 更新统计信息
        self.training_step += 1
        self.params_version += 1
        self.loss_history.append(loss.item())
        
        # 定期清理队列
        if self.training_step % 100 == 0:
            self._clean_queue()
        
        return {
            "status": "success",
            "step": self.training_step,
            "loss": loss.item(),
            "avg_loss": np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            "params_version": self.params_version,
            "queue_size": len(self.training_queue)
        }
    
    def _clean_queue(self):
        """清理训练队列"""
        current_size = len(self.training_queue)
        if current_size > self.config["clean_threshold"]:
            # 随机保留部分样本
            keep_indices = np.random.choice(
                current_size, 
                self.config["clean_keep"], 
                replace=False
            )
            new_queue = deque(
                [self.training_queue[i] for i in keep_indices], 
                maxlen=self.config["queue_size"]
            )
            self.training_queue = new_queue
            print(f"🧹 训练节点 {self.trainer_id}: 清理队列，从 {current_size} 减少到 {len(self.training_queue)}")
    
    def get_current_params(self) -> Dict[str, Any]:
        """获取当前模型参数（CPU版本，便于传输）"""
        params_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        return {
            "params": params_cpu,
            "version": self.params_version,
            "training_step": self.training_step,
            "total_samples": self.total_samples
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取训练节点状态"""
        return {
            "trainer_id": self.trainer_id,
            "training_step": self.training_step,
            "params_version": self.params_version,
            "queue_size": len(self.training_queue),
            "total_samples": self.total_samples,
            "avg_loss": np.mean(self.loss_history[-50:]) if self.loss_history else 0,
            "device": str(self.device)
        }