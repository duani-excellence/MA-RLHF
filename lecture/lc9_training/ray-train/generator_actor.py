# generator_actor.py
"""
生成节点Actor实现
"""

import ray
import torch
import numpy as np
import time
from typing import List, Dict, Any

from config import GENERATION_CONFIG, SIMULATION_CONFIG
from models import SharedLanguageModel
from data_utils import DataProcessor, TrainingSample


@ray.remote(num_gpus=0.5)
class GeneratorActor:
    """生成节点Actor：使用vLLM生成文本并发送给训练节点"""
    
    def __init__(
        self, 
        generator_id: str, 
        trainer_actor,
        device_id: int = 1,
        generation_config=None,
        simulation_config=None
    ):
        if generation_config is None:
            generation_config = GENERATION_CONFIG
        if simulation_config is None:
            simulation_config = SIMULATION_CONFIG
        
        self.generator_id = generator_id
        self.trainer_actor = trainer_actor
        self.config = generation_config
        self.simulation_config = simulation_config
        
        # 设置GPU
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        
        # 初始化本地模型副本
        self.local_model = SharedLanguageModel().to(self.device)
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(
            vocab_size=SIMULATION_CONFIG.get("simulate_vocab", 1000)
        )
        
        # 从训练节点获取初始参数
        self.params_version = 0
        self._update_from_trainer()
        
        # 统计信息
        self.generation_count = 0
        self.sent_samples = 0
        
        print(f"🚀 生成节点 {generator_id} 初始化完成，使用 {self.device}")
    
    def _update_from_trainer(self) -> bool:
        """从训练节点更新参数"""
        try:
            params_info = ray.get(self.trainer_actor.get_current_params.remote())
            
            # 加载参数到本地模型
            self.local_model.load_state_dict(params_info["params"])
            self.local_model.to(self.device)
            
            # 更新版本号
            self.params_version = params_info["version"]
            
            print(f"🔄 生成节点 {self.generator_id}: 参数更新到版本 {self.params_version}")
            return True
        
        except Exception as e:
            print(f"❌ 生成节点 {self.generator_id}: 更新参数失败 - {e}")
            return False
    
    def generate_with_vllm(
        self, 
        prompts: List[str], 
        max_tokens: int = None,
        temperature: float = None
    ) -> List[str]:
        """使用vLLM生成文本（模拟版本）"""
        if max_tokens is None:
            max_tokens = self.config["max_tokens"]
        if temperature is None:
            temperature = self.config["temperature"]
        
        # 切换到评估模式
        self.local_model.eval()
        
        generated_texts = []
        
        for prompt in prompts:
            try:
                # 模拟tokenization
                input_tokens = self.data_processor.simulate_tokenize(prompt)
                input_tensor = torch.tensor([input_tokens], device=self.device, dtype=torch.long)
                
                # 生成文本
                with torch.no_grad():
                    output_tokens = self.local_model.generate(
                        input_tensor, 
                        max_length=len(input_tokens) + max_tokens,
                        temperature=temperature
                    )
                
                # 解码回文本
                generated_text = self.data_processor.simulate_detokenize(
                    output_tokens[0].cpu().tolist()
                )
                generated_texts.append(generated_text)
            
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                generated_texts.append(f"[生成失败: {str(e)[:50]}]")
        
        self.generation_count += len(prompts)
        
        return generated_texts
    
    def prepare_training_data(
        self, 
        prompts: List[str], 
        generated_texts: List[str]
    ) -> List[Dict[str, Any]]:
        """准备训练数据"""
        # 创建训练样本
        samples = self.data_processor.create_training_samples(
            prompts, generated_texts, self.generator_id
        )
        
        # 转换为字典格式（便于序列化传输）
        data_dicts = []
        for sample in samples:
            data_dicts.append({
                "prompt": sample.prompt,
                "generated": sample.generated,
                "input_ids": sample.input_ids,
                "labels": sample.labels,
                "generator_id": sample.generator_id,
                "timestamp": sample.timestamp
            })
        
        return data_dicts
    
    def generate_and_send(
        self, 
        prompts: List[str], 
        batch_size: int = None
    ) -> Dict[str, Any]:
        """生成文本并发送给训练节点"""
        if batch_size is None:
            batch_size = self.config["generation_batch_size"]
        
        print(f"🎯 生成节点 {self.generator_id}: 开始生成 {len(prompts)} 个提示")
        
        all_samples = []
        
        # 分批处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # 使用vLLM生成文本
            generated_texts = self.generate_with_vllm(batch_prompts)
            
            # 准备训练数据
            batch_samples = self.prepare_training_data(batch_prompts, generated_texts)
            
            if batch_samples:
                try:
                    # 发送给训练节点
                    success = ray.get(
                        self.trainer_actor.receive_generated_data.remote(batch_samples)
                    )
                    
                    if success:
                        all_samples.extend(batch_samples)
                        self.sent_samples += len(batch_samples)
                        
                        # 显示生成结果示例
                        if i == 0 and batch_samples:
                            sample = batch_samples[0]
                            prompt_preview = sample["prompt"][:30] + "..." if len(sample["prompt"]) > 30 else sample["prompt"]
                            generated_preview = sample["generated"][:50] + "..." if len(sample["generated"]) > 50 else sample["generated"]
                            print(f"  📤 发送样本示例: '{prompt_preview}' → '{generated_preview}'")
                
                except Exception as e:
                    print(f"❌ 发送数据失败: {e}")
            
            # 定期更新参数
            if (i // batch_size) % self.config["update_frequency"] == 0:
                self._update_from_trainer()
        
        return {
            "generator_id": self.generator_id,
            "prompts_received": len(prompts),
            "samples_sent": len(all_samples),
            "total_generated": self.generation_count,
            "total_sent": self.sent_samples
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取生成节点状态"""
        return {
            "generator_id": self.generator_id,
            "generation_count": self.generation_count,
            "sent_samples": self.sent_samples,
            "params_version": self.params_version,
            "device": str(self.device)
        }