# Ray 

此lecture 掌握 Ray 的基础用法，以 复杂LLM Infra工程的关键功能特性：

1. 数据/Tensor 通信传输
2. PD分离
3. 训推分离



## 代码目录

该代码顺序待进一步整理和精简。

| 功能                                                    | 代码文件                            |      |
| ------------------------------------------------------- | ----------------------------------- | ---- |
| Ray 通信组件                                            | `actor_communication.py`            |      |
| Ray 传输整形列表                                        | `list_transfer.py`                  |      |
| Ray 传输 Tensor                                         | `tensor_transfer.py`                |      |
| Ray 维护共享队列（Cache），实现两个进程能够流式处理任务 | `shared_queue.py`                   |      |
| Ray 参数服务器梯度更新                                  | `parameter_server_pytorch.py`       |      |
| Ray 计算任务分离                                        | `ray_inferece_server_1.py`          |      |
|                                                         | `ray_inferece_server_2.py`          |      |
|                                                         | `ray_inferece_server_3.py`          |      |
| Ray 多节点训练                                          | `multi_node_training.py`            |      |
|                                                         | `ray_pytorch_training.py`           |      |
| Ray 多节点 LLM-RL 训练任务部署                          | `distributed_llm_system.py`         |      |
|                                                         | `distributed_llm_with_real_vllm.py` |      |

