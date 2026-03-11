# 本代码已实现

1. 实现分布式 PD 节点 和无限循环 step
2. 实现共享 scheduler, kvcache
3. 主函数实现 P节点、D节点、发送节点的持久化运行
4. PD 节点使用 ray 封装, 实现单控制器处理逻辑

# 笔记

1. Scheduler、KVCache 设计成分布式远程类对象, PD 节点可以共同访问远程对象, 特定场景需注意数据的读写安全
2. RayActorGroup 用于打包多个同类型的分布式actors
3. RayActorGroup 占多个 GPU, 对于群组的 Actor(model), 可以用 DeepSpeed ZeRO3 或标准的 Data Parallelism 进行初始化
4. RayActorGroup 被封装到推理节点 engine, engine 实现方法请求prompt读取、Forward、Cache更新操作
5. 单控制器逻辑: 在一个主函数中(CPU), 创建 Prefill/Decoding Engine, 这就是单控制器模式。 Engine 内 的 RayActorGroup 可以实现 SPMD（单程序多数据） 操作

# 其他
  
进一步：

在有本小节实现基础下, 可以类似 OpenRLHF 和 verl 框架实现 GRPO-RL 训练分离。

本节代码参考:

https://github.com/OpenRLHF/OpenRLHF