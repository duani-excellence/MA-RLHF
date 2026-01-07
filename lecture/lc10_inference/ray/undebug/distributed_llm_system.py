import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import deque

# 初始化Ray，设置GPU资源
ray.init(num_gpus=2, ignore_reinit_error=True)

print("=" * 60)
print("分布式LLM系统启动")
print("生成节点使用vLLM生成，训练节点异步训练")
print("=" * 60)

# ========== 1. 定义共享模型结构 ==========
class SharedLanguageModel(nn.Module):
    """共享的轻量级语言模型（简化版GPT）"""
    def __init__(self, vocab_size=50257, embedding_dim=256, hidden_dim=512, n_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 文本生成部分
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(1024, embedding_dim)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 创建位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 嵌入层
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        
        # 合并嵌入
        embeddings = self.dropout(token_embeds + pos_embeds)
        
        # 注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Transformer编码
        transformer_output = self.transformer(embeddings, src_key_padding_mask=~attention_mask.bool())
        
        # 输出logits
        logits = self.output_layer(transformer_output)
        
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=0.8):
        """简化的文本生成（实际中会用vLLM代替）"""
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length):
                # 前向传播
                logits = self.forward(generated)
                
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # 采样
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                
                # 拼接
                generated = torch.cat([generated, next_token], dim=-1)
                
                # 简单停止条件
                if next_token.item() == 50256:  # 假设的EOS token
                    break
            
            return generated

# ========== 2. 训练节点Actor ==========
@ray.remote(num_gpus=0.5)
class TrainerActor:
    def __init__(self, trainer_id="Trainer-1"):
        self.trainer_id = trainer_id
        
        # 设置GPU
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        
        # 初始化模型
        self.model = SharedLanguageModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练队列（存储生成节点发送的数据）
        self.training_queue = deque(maxlen=10000)
        self.training_step = 0
        self.params_version = 0
        
        # 记录信息
        self.total_samples = 0
        self.loss_history = []
        
        print(f"✅ 训练节点 {trainer_id} 初始化完成，使用 {self.device}")
    
    def receive_generated_data(self, batch_data: List[Dict]):
        """接收生成节点发送的数据"""
        if not batch_data:
            return False
        
        # 将数据添加到训练队列
        for data in batch_data:
            self.training_queue.append(data)
        
        new_samples = len(batch_data)
        self.total_samples += new_samples
        
        print(f"📥 训练节点 {self.trainer_id}: 收到 {new_samples} 个样本，队列大小: {len(self.training_queue)}")
        return True
    
    def train_step(self, batch_size=32):
        """执行一个训练步骤"""
        if len(self.training_queue) < batch_size:
            return {
                "status": "waiting",
                "queue_size": len(self.training_queue),
                "required": batch_size
            }
        
        # 从队列中采样一批数据
        indices = np.random.choice(len(self.training_queue), batch_size, replace=False)
        batch = [self.training_queue[i] for i in indices]
        
        # 准备训练数据
        input_ids = []
        labels = []
        
        for data in batch:
            input_ids.append(data["input_ids"])
            labels.append(data["labels"])
        
        # 转换为tensor
        input_tensor = torch.stack(input_ids).to(self.device)
        label_tensor = torch.stack(labels).to(self.device)
        
        # 训练步骤
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        logits = self.model(input_tensor)
        
        # 计算损失（仅对非padding部分）
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            label_tensor.view(-1)
        )
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新统计
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
            "avg_loss": np.mean(self.loss_history[-100:]),
            "params_version": self.params_version,
            "queue_size": len(self.training_queue)
        }
    
    def _clean_queue(self):
        """清理训练队列"""
        current_size = len(self.training_queue)
        if current_size > 5000:
            # 随机保留部分样本
            keep_indices = np.random.choice(current_size, 4000, replace=False)
            new_queue = deque([self.training_queue[i] for i in keep_indices], maxlen=10000)
            self.training_queue = new_queue
            print(f"🧹 训练节点 {self.trainer_id}: 清理队列，从 {current_size} 减少到 {len(self.training_queue)}")
    
    def get_current_params(self):
        """获取当前模型参数（CPU版本，便于传输）"""
        params_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return {
            "params": params_cpu,
            "version": self.params_version,
            "training_step": self.training_step,
            "total_samples": self.total_samples
        }
    
    def get_status(self):
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

# ========== 3. 生成节点Actor ==========
@ray.remote(num_gpus=0.5)
class GeneratorActor:
    def __init__(self, generator_id: str, trainer_actor, device_id=1):
        self.generator_id = generator_id
        self.trainer_actor = trainer_actor
        
        # 设置GPU
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        
        # 初始化本地模型副本
        self.local_model = SharedLanguageModel().to(self.device)
        
        # 从训练节点获取初始参数
        self._update_from_trainer()
        
        # 生成参数
        self.generation_count = 0
        self.sent_samples = 0
        
        # vLLM模拟参数
        self.vocab_size = 50257
        
        print(f"🚀 生成节点 {generator_id} 初始化完成，使用 {self.device}")
    
    def _update_from_trainer(self):
        """从训练节点更新参数"""
        try:
            params_info = ray.get(self.trainer_actor.get_current_params.remote())
            self.local_model.load_state_dict(params_info["params"])
            self.local_model.to(self.device)
            self.params_version = params_info["version"]
            
            print(f"🔄 生成节点 {self.generator_id}: 参数更新到版本 {self.params_version}")
            return True
        except Exception as e:
            print(f"❌ 生成节点 {self.generator_id}: 更新参数失败 - {e}")
            return False
    
    def generate_with_vllm(self, prompts: List[str], max_tokens=100, temperature=0.8):
        """使用vLLM生成文本（这里是模拟版本）"""
        self.local_model.eval()
        
        # 模拟vLLM生成 - 实际中这里会调用vLLM API
        generated_texts = []
        
        for prompt in prompts:
            # 模拟tokenization（简化）
            input_tokens = self._simulate_tokenize(prompt)
            input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
            
            # 生成文本
            with torch.no_grad():
                output_tokens = self.local_model.generate(
                    input_tensor, 
                    max_length=len(input_tokens) + max_tokens,
                    temperature=temperature
                )
            
            # 解码回文本
            generated_text = self._simulate_detokenize(output_tokens[0].cpu().tolist())
            generated_texts.append(generated_text)
        
        self.generation_count += len(prompts)
        
        return generated_texts
    
    def _simulate_tokenize(self, text: str) -> List[int]:
        """模拟tokenization（简化版）"""
        # 实际中应该使用真正的tokenizer
        return [ord(c) % 1000 for c in text[:50]] + [0]  # 简化
    
    def _simulate_detokenize(self, tokens: List[int]) -> str:
        """模拟detokenization（简化版）"""
        return ''.join(chr(t % 256) if t < 256 else '?' for t in tokens[:100])
    
    def create_training_samples(self, prompts: List[str], generated_texts: List[str]) -> List[Dict]:
        """从生成结果创建训练样本"""
        training_samples = []
        
        for prompt, generated in zip(prompts, generated_texts):
            # 模拟创建输入和目标
            combined = prompt + " " + generated
            
            # tokenize
            token_ids = self._simulate_tokenize(combined)
            if len(token_ids) < 5:
                continue
            
            # 创建训练样本
            input_ids = token_ids[:-1]
            labels = token_ids[1:]
            
            # 转换为tensor
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            label_tensor = torch.tensor(labels, dtype=torch.long)
            
            # 添加到训练样本
            training_samples.append({
                "prompt": prompt,
                "generated": generated,
                "input_ids": input_tensor,
                "labels": label_tensor,
                "generator_id": self.generator_id,
                "timestamp": time.time()
            })
        
        return training_samples
    
    def generate_and_send(self, prompts: List[str], batch_size=4):
        """生成文本并发送给训练节点"""
        print(f"🎯 生成节点 {self.generator_id}: 开始生成 {len(prompts)} 个提示")
        
        # 分批生成
        all_samples = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # 使用vLLM生成文本
            generated_texts = self.generate_with_vllm(batch_prompts)
            
            # 创建训练样本
            batch_samples = self.create_training_samples(batch_prompts, generated_texts)
            
            if batch_samples:
                # 发送给训练节点
                success = ray.get(self.trainer_actor.receive_generated_data.remote(batch_samples))
                
                if success:
                    all_samples.extend(batch_samples)
                    self.sent_samples += len(batch_samples)
                    
                    # 显示生成结果示例
                    if i == 0 and batch_samples:
                        sample = batch_samples[0]
                        print(f"  📤 发送样本示例: '{sample['prompt'][:30]}...' → '{sample['generated'][:50]}...'")
            
            # 定期更新参数（每生成5批更新一次）
            if (i // batch_size) % 5 == 0:
                self._update_from_trainer()
        
        return {
            "generator_id": self.generator_id,
            "prompts_received": len(prompts),
            "samples_sent": len(all_samples),
            "total_generated": self.generation_count,
            "total_sent": self.sent_samples
        }
    
    def get_status(self):
        """获取生成节点状态"""
        return {
            "generator_id": self.generator_id,
            "generation_count": self.generation_count,
            "sent_samples": self.sent_samples,
            "params_version": self.params_version,
            "device": str(self.device)
        }

# ========== 4. 协调器 ==========
@ray.remote
class SystemCoordinator:
    def __init__(self, num_generators=2):
        self.num_generators = num_generators
        
        # 创建训练节点
        print("🤖 创建训练节点...")
        self.trainer = TrainerActor.remote()
        
        # 创建生成节点
        print(f"🤖 创建 {num_generators} 个生成节点...")
        self.generators = []
        for i in range(num_generators):
            generator = GeneratorActor.remote(
                generator_id=f"Generator-{i+1}",
                trainer_actor=self.trainer,
                device_id=i % 2  # 轮流分配到不同GPU
            )
            self.generators.append(generator)
        
        # 训练状态
        self.is_training = False
        self.is_generating = False
        
        print(f"✅ 系统协调器初始化完成: 1个训练节点 + {num_generators}个生成节点")
    
    def start_training_loop(self, interval=2.0):
        """启动训练循环"""
        self.is_training = True
        
        @ray.remote
        def training_worker(trainer, interval):
            while True:
                # 执行训练步骤
                result = ray.get(trainer.train_step.remote())
                
                if result["status"] == "success":
                    step = result["step"]
                    loss = result["loss"]
                    
                    if step % 10 == 0:
                        print(f"🏋️‍♂️ 训练步骤 {step}: 损失={loss:.4f}, 队列={result['queue_size']}")
                
                time.sleep(interval)
        
        # 启动训练工作线程
        self.training_future = training_worker.remote(self.trainer, interval)
        print("🏋️‍♂️ 训练循环已启动")
    
    def start_generation_loop(self, prompts_file="prompts.txt", interval=3.0):
        """启动生成循环"""
        self.is_generating = True
        
        # 加载或生成提示词
        prompts = self._load_prompts(prompts_file)
        
        @ray.remote
        def generation_worker(generator, prompts, interval):
            idx = 0
            while True:
                # 每次取一批提示词
                batch_prompts = prompts[idx:idx+2]
                idx = (idx + 2) % len(prompts)
                
                if batch_prompts:
                    # 生成并发送
                    result = ray.get(generator.generate_and_send.remote(batch_prompts))
                    
                    print(f"🎨 {result['generator_id']}: 发送 {result['samples_sent']} 个样本")
                
                time.sleep(interval)
        
        # 为每个生成节点启动工作线程
        self.generation_futures = []
        for generator in self.generators:
            future = generation_worker.remote(generator, prompts, interval + np.random.uniform(0, 1))
            self.generation_futures.append(future)
        
        print(f"🎨 生成循环已启动，{len(self.generators)} 个生成节点工作")
    
    def _load_prompts(self, filename):
        """加载提示词文件"""
        # 如果没有文件，生成一些示例提示词
        example_prompts = [
            "Once upon a time",
            "The future of AI",
            "In a distant galaxy",
            "The secret to happiness",
            "How to learn programming",
            "The meaning of life",
            "A story about a robot",
            "The benefits of exercise",
            "Climate change solutions",
            "Artificial intelligence ethics",
            "The history of computing",
            "Machine learning applications",
            "Deep learning techniques",
            "Natural language processing",
            "Computer vision challenges",
            "Reinforcement learning basics",
            "Neural network architectures",
            "Data science workflow",
            "Big data analytics",
            "Cloud computing advantages"
        ]
        
        try:
            with open(filename, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except:
            prompts = example_prompts
            print(f"📝 使用示例提示词 ({len(prompts)} 个)")
        
        return prompts
    
    def get_system_status(self):
        """获取系统状态"""
        trainer_status = ray.get(self.trainer.get_status.remote())
        
        generator_statuses = []
        for generator in self.generators:
            status = ray.get(generator.get_status.remote())
            generator_statuses.append(status)
        
        return {
            "trainer": trainer_status,
            "generators": generator_statuses,
            "is_training": self.is_training,
            "is_generating": self.is_generating
        }
    
    def stop_system(self):
        """停止系统"""
        self.is_training = False
        self.is_generating = False
        
        print("🛑 停止系统中...")
        
        # 获取最终状态
        status = self.get_system_status()
        
        print("\n" + "=" * 60)
        print("最终系统统计:")
        print(f"训练步骤: {status['trainer']['training_step']}")
        print(f"总样本数: {status['trainer']['total_samples']}")
        print(f"平均损失: {status['trainer']['avg_loss']:.4f}")
        
        for gen in status['generators']:
            print(f"{gen['generator_id']}: 生成 {gen['generation_count']} 次, 发送 {gen['sent_samples']} 样本")
        
        print("=" * 60)
        return status

# ========== 5. 主程序 ==========
def main():
    print("🚀 启动分布式LLM训练系统")
    print("=" * 60)
    
    # 创建协调器
    coordinator = SystemCoordinator.remote(num_generators=2)
    
    # 启动训练循环
    ray.get(coordinator.start_training_loop.remote(interval=1.5))
    
    # 启动生成循环
    ray.get(coordinator.start_generation_loop.remote(interval=2.0))
    
    print("\n✅ 系统启动完成！")
    print("系统将运行30秒，然后显示统计信息...")
    
    # 运行一段时间
    try:
        for i in range(30):
            if i % 10 == 0:
                # 每10秒显示一次状态
                status = ray.get(coordinator.get_system_status.remote())
                print(f"\n⏱️ 运行 {i} 秒后状态:")
                print(f"  训练步骤: {status['trainer']['training_step']}")
                print(f"  训练队列: {status['trainer']['queue_size']}")
                print(f"  生成节点: {len(status['generators'])} 个活跃")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    
    # 停止系统并获取最终状态
    final_status = ray.get(coordinator.stop_system.remote())
    
    # 保存最终参数
    trainer_params = ray.get(coordinator.trainer.get_current_params.remote())
    print(f"\n💾 最终模型参数版本: {trainer_params['version']}")
    
    # 关闭Ray
    ray.shutdown()
    print("\n🎉 系统运行完成！")

if __name__ == "__main__":
    main()