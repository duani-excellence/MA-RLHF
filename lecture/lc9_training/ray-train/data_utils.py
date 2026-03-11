# data_utils.py
"""
数据处理和工具函数
"""

import numpy as np
import torch
from typing import List, Dict, Any
from dataclasses import dataclass

from config import SIMULATION_CONFIG


@dataclass
class TrainingSample:
    """训练样本数据结构"""
    prompt: str
    generated: str
    input_ids: torch.Tensor
    labels: torch.Tensor
    generator_id: str
    timestamp: float


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
    
    def simulate_tokenize(self, text: str, max_length=50) -> List[int]:
        """模拟tokenization（简化版）"""
        # 实际中应该使用真正的tokenizer
        tokens = []
        for i, char in enumerate(text[:max_length]):
            token_id = ord(char) % self.vocab_size
            tokens.append(token_id)
        
        # 添加EOS token
        tokens.append(self.vocab_size - 1)
        
        return tokens
    
    def simulate_detokenize(self, tokens: List[int]) -> str:
        """模拟detokenization（简化版）"""
        text_chars = []
        for token in tokens[:100]:
            if token < 256:
                text_chars.append(chr(token))
            else:
                text_chars.append('?')
        
        return ''.join(text_chars)
    
    def create_training_samples(
        self, 
        prompts: List[str], 
        generated_texts: List[str],
        generator_id: str
    ) -> List[TrainingSample]:
        """从生成结果创建训练样本"""
        training_samples = []
        
        for prompt, generated in zip(prompts, generated_texts):
            # 模拟创建输入和目标
            combined = prompt + " " + generated
            
            # tokenize
            token_ids = self.simulate_tokenize(combined)
            if len(token_ids) < 5:
                continue
            
            # 创建训练样本
            input_ids = token_ids[:-1]
            labels = token_ids[1:]
            
            # 转换为tensor
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            label_tensor = torch.tensor(labels, dtype=torch.long)
            
            # 创建样本对象
            sample = TrainingSample(
                prompt=prompt,
                generated=generated,
                input_ids=input_tensor,
                labels=label_tensor,
                generator_id=generator_id,
                timestamp=np.random.random()  # 模拟时间戳
            )
            
            training_samples.append(sample)
        
        return training_samples
    
    def batch_samples(self, samples: List[TrainingSample], batch_size: int):
        """将样本分批"""
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]


class PromptManager:
    """提示词管理器"""
    
    @staticmethod
    def load_prompts(filename="prompts.txt"):
        """加载提示词文件"""
        example_prompts = [
            "Once upon a time in a distant galaxy",
            "The future of artificial intelligence",
            "In the year 2050, robots have become",
            "The secret to happiness is",
            "How to learn programming effectively",
            "The meaning of life according to AI",
            "A story about a robot who learned to love",
            "The benefits of regular exercise include",
            "Climate change solutions that work",
            "Artificial intelligence ethics guidelines",
            "The history of computing from abacus to quantum",
            "Machine learning applications in healthcare",
            "Deep learning techniques for image recognition",
            "Natural language processing challenges",
            "Computer vision in autonomous vehicles",
            "Reinforcement learning for game playing",
            "Neural network architectures comparison",
            "Data science workflow from data to insights",
            "Big data analytics for business intelligence",
            "Cloud computing advantages and disadvantages"
        ]
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            prompts = example_prompts
            print(f"📝 提示词文件未找到，使用 {len(prompts)} 个示例提示词")
        
        return prompts