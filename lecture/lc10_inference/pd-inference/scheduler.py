# scheduler
# 调度简单，异步接收请求
# 获取 wating 列表
# 获取 running 列表
# 获取 buffer 列表 ( prefill 后压到 buffer 里，未被 decoding 节点所取的请求 )

from queue import deque
import ray
from typing import Dict, List, Set, Tuple, Optional, Any

from request import Request

from dataclasses import dataclass, field


@dataclass
class SchedulerInfo:
    ids: List[int] = field(default_factory=list)
    prompts: List[List[str]] = field(default_factory=list)

    kv_len: List[int] = field(default_factory=list)
    last_pos: List[int] = field(default_factory=list)

    decoding_batch: int = 0
    prefill_batch: int = 0


@ray.remote()
class Scheduler:
    """管理所有请求的调度和状态"""

    def __init__(self, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len
        self.requests = {}  # request_id -> Request
        self.waiting_queue = deque()
        self.running_requests = set()

    async def add_request(self, prompt: List[int], max_seq_len: int) -> int:
        """添加新请求,返回请求ID"""
        request_id = len(self.requests)
        request = Request(request_id, prompt, max_seq_len)
        self.requests[request_id] = request
        self.waiting_queue.append(request_id)
        return request_id

    def get_available_request(self) -> int:
        """获取可用的批次空位数量"""
        return len(self.waiting_queue) + len(self.running_requests)

    def get_waiting_requests(self,):
        """"""
        if len(self.waiting_queue) == 0:
            return None

        info = SchedulerInfo()

        for _ in range(len(self.waiting_queue)):
            request_id = self.waiting_queue.popleft()
            prompt = self.requests[request_id].prompt
            info.prompts.append(prompt)
            info.ids.append(request_id)
            info.last_pos.append(len(prompt))
            info.prefill_batch += 1
        return info

    def get_running_requests(self,):
        """"""
        if len(self.running_requests) == 0:
            return None

        info = SchedulerInfo()

        for request_id in self.running_requests:
            # request_id = self.waiting_queue.popleft()
            prompt = self.requests[request_id].generated_tokens[-1]
            info.prompts.append(prompt)
            info.ids.append(request_id)
            info.last_pos.append(
                len(prompt)+len(self.requests[request_id].generated_tokens))
            info.decoding_batch += 1
        return info

    def update_request(self, request_id: int, next_token: int):
        """更新请求状态"""
        if request_id in self.requests:
            request = self.requests[request_id]
            request.add_token(next_token)

            if request_id not in self.running_requests:
                self.running_requests.add(request_id)

            if request.is_finished():
                self.running_requests.discard(request_id)

    def has_pending_requests(self) -> bool:
        """检查是否有未完成的请求"""
        return len(self.waiting_queue) > 0 or len(self.running_requests) > 0

    def get_num_pending_requests(self) -> int:
        return len(self.waiting_queue)

    def get_num_running_requests(self) -> int:
        return len(self.running_requests)

    def get_running_request_ids(self) -> List[int]:
        """获取当前正在运行的请求ID"""
        return list(self.running_requests)
