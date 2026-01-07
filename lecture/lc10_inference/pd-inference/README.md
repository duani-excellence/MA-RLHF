本代码实现目的

1. 实现分布式 PD 节点 和无限循环 step
2. 实现共享 scheduler, kvcache
3. 主函数实现 P节点、D节点、发送节点的持久化运行
4. PD 节点使用 ray 封装, 实现单控制器处理逻辑
  
进一步：

在有本小节实现基础下, 可以类似 OpenRLHF 和 verl 框架实现 GRPO-RL 训练分离。

本节代码参考:

https://github.com/OpenRLHF/OpenRLHF