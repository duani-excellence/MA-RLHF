# coordinator.py
"""
系统协调器：管理训练和生成节点
"""

import ray
import numpy as np
import time
from typing import List, Dict, Any

from config import SYSTEM_CONFIG
from data_utils import PromptManager
from trainer_actor import TrainerActor
from generator_actor import GeneratorActor


@ray.remote
class SystemCoordinator:
    """系统协调器：创建和管理整个分布式系统"""
    
    def __init__(self, num_generators=None, system_config=None):
        if system_config is None:
            system_config = SYSTEM_CONFIG
        if num_generators is None:
            num_generators = system_config["num_generators"]
        
        self.system_config = system_config
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
        
        # 系统状态
        self.is_training = False
        self.is_generating = False
        self.training_future = None
        self.generation_futures = []
        
        print(f"✅ 系统协调器初始化完成: 1个训练节点 + {num_generators}个生成节点")
    
    def start_training_loop(self, interval=None):
        """启动训练循环"""
        if interval is None:
            interval = self.system_config["train_interval"]
        
        self.is_training = True
        
        @ray.remote
        def training_worker(trainer, interval):
            """训练工作线程"""
            while True:
                try:
                    # 执行训练步骤
                    result = ray.get(trainer.train_step.remote())
                    
                    if result["status"] == "success":
                        step = result["step"]
                        loss = result["loss"]
                        
                        # 定期显示训练状态
                        if step % 10 == 0:
                            print(f"🏋️‍♂️ 训练步骤 {step}: 损失={loss:.4f}, 队列={result['queue_size']}")
                
                except Exception as e:
                    print(f"❌ 训练循环错误: {e}")
                
                time.sleep(interval)
        
        # 启动训练工作线程
        self.training_future = training_worker.remote(self.trainer, interval)
        print("🏋️‍♂️ 训练循环已启动")
    
    def start_generation_loop(self, prompts_file="prompts.txt", interval=None):
        """启动生成循环"""
        if interval is None:
            interval = self.system_config["generate_interval"]
        
        self.is_generating = True
        
        # 加载提示词
        prompts = PromptManager.load_prompts(prompts_file)
        
        @ray.remote
        def generation_worker(generator, prompts, interval, worker_id):
            """生成工作线程"""
            idx = 0
            cycle_count = 0
            
            while True:
                try:
                    # 每次取一批提示词
                    batch_size = 2
                    start_idx = idx
                    end_idx = idx + batch_size
                    
                    if end_idx > len(prompts):
                        # 循环到开头
                        batch_prompts = prompts[start_idx:] + prompts[:end_idx - len(prompts)]
                        idx = end_idx - len(prompts)
                    else:
                        batch_prompts = prompts[start_idx:end_idx]
                        idx = end_idx % len(prompts)
                    
                    if batch_prompts:
                        # 生成并发送
                        result = ray.get(generator.generate_and_send.remote(batch_prompts))
                        
                        cycle_count += 1
                        if cycle_count % 3 == 0:
                            print(f"🎨 {result['generator_id']}(Worker-{worker_id}): "
                                  f"已发送 {result['samples_sent']} 个样本，总共 {result['total_sent']}")
                
                except Exception as e:
                    print(f"❌ 生成循环错误: {e}")
                
                time.sleep(interval + np.random.uniform(0, 0.5))
        
        # 为每个生成节点启动工作线程
        self.generation_futures = []
        for i, generator in enumerate(self.generators):
            future = generation_worker.remote(
                generator, prompts, interval + np.random.uniform(0, 1), i
            )
            self.generation_futures.append(future)
        
        print(f"🎨 生成循环已启动，{len(self.generators)} 个生成节点工作")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            trainer_status = ray.get(self.trainer.get_status.remote())
        except Exception as e:
            trainer_status = {"error": str(e)}
        
        generator_statuses = []
        for generator in self.generators:
            try:
                status = ray.get(generator.get_status.remote())
                generator_statuses.append(status)
            except Exception as e:
                generator_statuses.append({"error": str(e)})
        
        return {
            "trainer": trainer_status,
            "generators": generator_statuses,
            "is_training": self.is_training,
            "is_generating": self.is_generating,
            "num_generators": len(self.generators)
        }
    
    def stop_system(self) -> Dict[str, Any]:
        """停止系统并获取最终状态"""
        self.is_training = False
        self.is_generating = False
        
        print("🛑 停止系统中...")
        time.sleep(2)  # 等待正在进行的操作完成
        
        # 获取最终状态
        try:
            status = self.get_system_status()
            
            print("\n" + "=" * 60)
            print("最终系统统计:")
            
            if "error" not in status["trainer"]:
                trainer = status["trainer"]
                print(f"训练步骤: {trainer.get('training_step', 0)}")
                print(f"总样本数: {trainer.get('total_samples', 0)}")
                print(f"平均损失: {trainer.get('avg_loss', 0):.4f}")
                print(f"参数版本: {trainer.get('params_version', 0)}")
            else:
                print(f"训练节点错误: {status['trainer']['error']}")
            
            print("\n生成节点统计:")
            for i, gen in enumerate(status["generators"]):
                if "error" not in gen:
                    print(f"  {gen.get('generator_id', f'Generator-{i}')}: "
                          f"生成 {gen.get('generation_count', 0)} 次, "
                          f"发送 {gen.get('sent_samples', 0)} 样本")
                else:
                    print(f"  生成节点 {i} 错误: {gen['error']}")
            
            print("=" * 60)
        
        except Exception as e:
            print(f"❌ 获取最终状态失败: {e}")
            status = {"error": str(e)}
        
        return status