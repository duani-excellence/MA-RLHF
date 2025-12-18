import ray
import torch
import torch.nn as nn
from typing import List, Dict
import time
from transformers import AutoTokenizer

# 需要先安装: pip install vllm transformers

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM不可用，使用模拟模式")

@ray.remote(num_gpus=0.8)
class RealVLLMGenerator:
    """使用真实vLLM的生成节点"""
    
    def __init__(self, generator_id: str, trainer_actor, model_name="gpt2"):
        self.generator_id = generator_id
        self.trainer_actor = trainer_actor
        
        # 初始化vLLM模型
        if VLLM_AVAILABLE:
            print(f"🔄 生成节点 {generator_id}: 加载vLLM模型 {model_name}...")
            self.llm = LLM(model=model_name, tensor_parallel_size=1)
            self.sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100
            )
        else:
            self.llm = None
            print(f"⚠️  生成节点 {generator_id}: vLLM不可用，使用模拟模式")
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 统计
        self.generation_count = 0
        
        print(f"✅ 真实vLLM生成节点 {generator_id} 初始化完成")
    
    def generate_with_real_vllm(self, prompts: List[str]):
        """使用真实vLLM生成文本"""
        if self.llm is None:
            # 模拟生成
            return [f"模拟生成: {p[:20]}..." for p in prompts]
        
        try:
            # 使用vLLM生成
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            generated_texts = []
            for output in outputs:
                generated = output.outputs[0].text
                generated_texts.append(generated)
            
            self.generation_count += len(prompts)
            return generated_texts
        
        except Exception as e:
            print(f"❌ vLLM生成失败: {e}")
            return []
    
    def prepare_training_data(self, prompts: List[str], generated_texts: List[str]):
        """准备训练数据"""
        training_samples = []
        
        for prompt, generated in zip(prompts, generated_texts):
            # 组合提示和生成文本
            combined_text = prompt + generated
            
            # Tokenize
            encoding = self.tokenizer(
                combined_text,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors="pt"
            )
            
            # 创建训练样本
            input_ids = encoding["input_ids"][0]
            
            # 创建labels（与input_ids相同，用于语言模型训练）
            labels = input_ids.clone()
            
            # 可选：对prompt部分设置ignore_index
            prompt_length = len(self.tokenizer.encode(prompt))
            labels[:prompt_length] = -100
            
            training_samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": encoding["attention_mask"][0],
                "prompt": prompt,
                "generated": generated
            })
        
        return training_samples
    
    def generate_and_send_batch(self, prompts: List[str]):
        """生成一批数据并发送给训练节点"""
        print(f"🎨 {self.generator_id}: 使用vLLM生成 {len(prompts)} 个提示")
        
        # 1. 生成文本
        generated_texts = self.generate_with_real_vllm(prompts)
        
        if not generated_texts:
            return {"status": "failed", "reason": "生成失败"}
        
        # 2. 准备训练数据
        training_samples = self.prepare_training_data(prompts, generated_texts)
        
        if not training_samples:
            return {"status": "failed", "reason": "数据准备失败"}
        
        # 3. 发送给训练节点
        try:
            success = ray.get(self.trainer_actor.receive_generated_data.remote(training_samples))
            
            if success:
                print(f"📤 {self.generator_id}: 成功发送 {len(training_samples)} 个样本")
                
                # 显示示例
                if len(training_samples) > 0:
                    sample = training_samples[0]
                    print(f"  示例: '{sample['prompt'][:30]}...'")
                    print(f"  生成: '{sample['generated'][:50]}...'")
                
                return {
                    "status": "success",
                    "samples_sent": len(training_samples),
                    "generator_id": self.generator_id
                }
        
        except Exception as e:
            print(f"❌ 发送失败: {e}")
        
        return {"status": "failed", "reason": "发送失败"}

# 使用示例
def main_real_vllm():
    """使用真实vLLM的主程序"""
    ray.init(num_gpus=2)
    
    print("🚀 启动真实vLLM分布式系统")
    
    # 创建训练节点
    from distributed_llm_system import TrainerActor  # 导入之前的训练节点
    
    trainer = TrainerActor.remote()
    
    # 创建真实vLLM生成节点
    generator = RealVLLMGenerator.remote(
        generator_id="RealVLLM-Generator",
        trainer_actor=trainer,
        model_name="gpt2"  # 可以换成更大的模型
    )
    
    # 测试生成
    test_prompts = [
        "Artificial intelligence is",
        "The future of machine learning",
        "Deep learning models can"
    ]
    
    print("\n🧪 测试vLLM生成...")
    result = ray.get(generator.generate_and_send_batch.remote(test_prompts))
    print(f"测试结果: {result}")
    
    # 启动训练
    print("\n🏋️‍♂️ 启动训练...")
    for i in range(5):
        train_result = ray.get(trainer.train_step.remote(batch_size=2))
        print(f"训练步骤 {i}: {train_result}")
        time.sleep(1)
    
    ray.shutdown()
    print("\n✅ 真实vLLM系统测试完成")

if __name__ == "__main__":
    if VLLM_AVAILABLE:
        main_real_vllm()
    else:
        print("❌ vLLM不可用，请先安装: pip install vllm")
        # 运行模拟版本
        import distributed_llm_system
        distributed_llm_system.main()