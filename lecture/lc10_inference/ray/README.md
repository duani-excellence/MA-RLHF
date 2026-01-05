# Ray 

此 lecture 掌握 Ray 的基础用法，以 复杂 LLM Infra工程的关键功能特性：

1. 数据/Tensor 通信传输
2. PD分离
3. 训推分离

代码由 AI 生成, 正确性未经过验证

## 代码目录

该代码顺序待进一步整理和精简。

| 功能                                                    | 代码文件                            | 备注                |
| ------------------------------------------------------- | ----------------------------------- | ------------------- |
| Ray 通信组件                                            | `actor_communication.py`            |                     |
| Ray 传输整形列表                                        | `list_transfer.py`                  |                     |
| Ray 传输 Tensor                                         | `tensor_transfer.py`                |                     |
| Ray 维护共享队列（Cache），实现两个进程能够流式处理任务 | `shared_queue.py`                   |                     |
| Ray 参数服务器梯度更新                                  | `parameter_server_pytorch.py`       |                     |
| Ray 计算任务分离                                        | `ray_inferece_server_1.py`          | 异步PD              |
|                                                         | `ray_inferece_server_2.py`          |                     |
|                                                         | `ray_inferece_server_3.py`          | 异步PD主体订阅      |
| Ray 多节点训练                                          | `multi_node_training.py`            | 训推分离            |
|                                                         | `ray_pytorch_training.py`           | 训推分离+参数服务器 |
|                                                         | `ray_pytorch_training_gpu.py`       | 训推分离+GPU部署    |
| Ray 多节点 LLM-RL 训练任务部署                          | `distributed_llm_system.py`         | 训推节点loop        |
|                                                         | `distributed_llm_with_real_vllm.py` | 训推+vllm           |
| Ray + torch.distbuted + Gloo backend                    | `ray_torch_distributed_init.py`     | ray 群组初始化      |
|                                                         | `ray_torch_distributed_step.py`     | ray 群组持久化运行  |


## 手写 ray debug 版本


| 功能                                                    | 代码文件                   | 备注                             |
| ------------------------------------------------------- | -------------------------- | -------------------------------- |
| Ray 通信组件                                            | `actor_communication_1.py` | 发收同步                         |
|                                                         | `actor_communication_2.py` | 收异步                           |
|                                                         | `actor_communication_2.py` | 收发异步                         |
| Ray 维护共享队列（Cache），实现两个进程能够流式处理任务 | `shared_queue_1.py`        | 通过一个可访问的共享队列实现通信 |
| Ray 远程函数                                            | `remote_function.py`       | 不同设备数据处理, 手动to         |
|                                                         | `remote_function_1.py`     | 不同设备数据处理, gpu            |
| Ray tensor 传输                                         | `tensor_transfer_1.py`     | 数据存储在共享内存中，使用零拷贝 |
| Ray 分布式操作                                          | `ray_all_reduce_mean.py`   | 实现规约操作                     |

