本代码实现目的

1. 掌握 PD 分离基础实现
   - 节点初始化可指定设备（以 CPU 实现调试为主，可选实现 GPU）
   - 单 P 节点和单 D 节点
   - 异步 KVCache 传输，而 D 节点主要负责 cache 的管理
2. 掌握 Ray 实现复杂推理系统，实现feature有：
    - 单 P 节点和多 D worker
    - KVCache 可以中心化，如用 NVme 设备进行 offload 存储，decoding需要时，异步拉取，更加 pratical
  
进一步：

在有本小节实现基础下, 可以类似 OpenRLHF 和 verl 框架实现训练分离。

