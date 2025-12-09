from queue import deque
from typing import Dict, List, Set, Tuple, Optional, Any

from .request import Request


class Schedular:
    """管理所有请求的调度和状态"""

    def __init__(self, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len
        self.requests = {}  # request_id -> Request
        self.waiting_queue = deque()
        self.running_requests = set()

    def add_request(self, prompt: List[int], max_seq_len: int) -> int:
        """添加新请求,返回请求ID"""
        request_id = len(self.requests)
        request = Request(request_id, prompt, max_seq_len)
        self.requests[request_id] = request
        self.waiting_queue.append(request_id)
        return request_id

    def get_available_request(self) -> int:
        """获取可用的批次空位数量"""
        return len(self.waiting_queue)

    def get_pending_requests(self, max_count: int) -> List[Tuple[int, List[int]]]:
        """获取等待处理的请求"""
        available_slots = self.get_available_request()
        count = min(max_count, available_slots, len(self.waiting_queue))

        requests_to_process = []
        for _ in range(count):
            if not self.waiting_queue:
                break
            request_id = self.waiting_queue.popleft()
            request = self.requests[request_id]
            request.status = "REQUEST_RUNNING"
            self.running_requests.add(request_id)
            requests_to_process.append((request_id, request.prompt))
        return requests_to_process

    def update_request(self, request_id: int, next_token: int):
        """更新请求状态"""
        if request_id in self.requests:
            request = self.requests[request_id]
            request.add_token(next_token)
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
