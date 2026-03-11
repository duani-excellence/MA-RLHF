# main.py
"""
主程序入口
"""

import ray
import time
import sys

from config import SYSTEM_CONFIG
from coordinator import SystemCoordinator


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 启动分布式LLM训练系统")
    print("系统架构: 生成节点(vLLM) → 训练节点 → 参数同步 → 生成节点")
    print("=" * 60)
    
    # 初始化Ray
    try:
        ray.init(
            num_gpus=2, 
            ignore_reinit_error=True,
            include_dashboard=False,
            logging_level="INFO"
        )
        print(f"✅ Ray初始化成功")
        print(f"  可用资源: {ray.available_resources()}")
    except Exception as e:
        print(f"❌ Ray初始化失败: {e}")
        sys.exit(1)
    
    try:
        # 创建协调器
        coordinator = SystemCoordinator.remote(
            num_generators=SYSTEM_CONFIG["num_generators"],
            system_config=SYSTEM_CONFIG
        )
        
        # 启动训练循环
        print("\n启动训练循环...")
        ray.get(coordinator.start_training_loop.remote(
            interval=SYSTEM_CONFIG["train_interval"]
        ))
        
        # 启动生成循环
        print("\n启动生成循环...")
        ray.get(coordinator.start_generation_loop.remote(
            prompts_file="prompts.txt",
            interval=SYSTEM_CONFIG["generate_interval"]
        ))
        
        print("\n✅ 系统启动完成！")
        print(f"系统将运行 {SYSTEM_CONFIG['runtime_seconds']} 秒...")
        
        # 运行一段时间
        runtime = SYSTEM_CONFIG["runtime_seconds"]
        status_interval = SYSTEM_CONFIG["status_interval"]
        
        for i in range(runtime):
            if i % status_interval == 0:
                # 定期显示状态
                status = ray.get(coordinator.get_system_status.remote())
                
                print(f"\n⏱️ 运行 {i} 秒后状态:")
                
                trainer = status.get("trainer", {})
                if "error" not in trainer:
                    print(f"  训练步骤: {trainer.get('training_step', 0)}")
                    print(f"  训练队列: {trainer.get('queue_size', 0)}")
                    print(f"  平均损失: {trainer.get('avg_loss', 0):.4f}")
                else:
                    print(f"  训练节点: {trainer['error']}")
                
                print(f"  生成节点: {len(status.get('generators', []))} 个活跃")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断，正在停止系统...")
    
    finally:
        # 停止系统并获取最终状态
        print("\n正在停止系统...")
        final_status = ray.get(coordinator.stop_system.remote())
        
        # 关闭Ray
        ray.shutdown()
        
        print("\n" + "=" * 60)
        print("🎉 系统运行完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()