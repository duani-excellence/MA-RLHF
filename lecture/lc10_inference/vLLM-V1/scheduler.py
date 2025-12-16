from queue import deque
from typing import Dict, List, Set, Tuple, Optional, Any

from request import Request

from dataclasses import dataclass, field


@dataclass
class SchedulerInfo:
    ids: List[int] = field(default_factory=list)
    chunk_prompts: List[List[int]] = field(default_factory=list)

    chunk_idx: List[int] = field(default_factory=list)
    chunk_len: List[int] = field(default_factory=list)

    merge_prompt: List[int] = field(default_factory=list)

    kv_len: List[int] = field(default_factory=list)
    kv_page_len: List[int] = field(default_factory=list)

    is_decoding: List[bool] = field(default_factory=list)

    last_pos: List[int] = field(default_factory=list)

    decoding_batch: int = 0
    prefill_batch: int = 0


class Scheduler:
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
        return len(self.waiting_queue) + len(self.running_requests)

    def get_requests(self,
                     max_batch_tokens: int = 8192,
                     max_decoding_batch: int = 0,
                     max_prefill_batch: int = 100,):
        """"""

        info = SchedulerInfo()

        # 将所有请求至为运行状态
        for _ in range(len(self.waiting_queue)):
            request_id = self.waiting_queue.popleft()
            request = self.requests[request_id]
            request.status = "REQUEST_RUNNING"
            self.running_requests.add(request_id)

        # 处理 decoding 请求
        count_batch_token = 0
        for i in self.running_requests:
            # decoding batching
            req = self.requests[i]
            if req.is_decoding():
                info.ids.append(i)
                info.chunk_prompts.append(
                    [req.generated_tokens[-1]]
                )
                info.kv_len.append(req.kv_len)
                info.last_pos.append(0)

                info.chunk_idx.append(count_batch_token)
                info.chunk_len.append(1)

                info.is_decoding.append(True)
                count_batch_token += 1
            if count_batch_token == max_decoding_batch:
                break
        info.decoding_batch = count_batch_token

        # 处理 prefill 请求
        prefill_batch_count = 0
        for i in self.running_requests:
            req = self.requests[i]
            if req.is_decoding():
                continue

            info.ids.append(i)
            info.kv_len.append(req.kv_len)

            avalable_tokens = max_batch_tokens - count_batch_token
            start = req.kv_len
            if len(req.prompt[start:]) <= avalable_tokens:
                # 当前请求数据可以prefill
                info.chunk_prompts.append(req.prompt[start:])
                info.last_pos.append(len(req.prompt[start:])-1)

                info.chunk_idx.append(count_batch_token)
                info.chunk_len.append(len(req.prompt[start:]))
                count_batch_token += len(req.prompt[start:])

            else:
                # 当前请求数据可以chunked-prefill
                info.chunk_prompts.append(
                    req.prompt[start: start+avalable_tokens])
                info.last_pos.append(-1)

                info.chunk_idx.append(count_batch_token)
                info.chunk_len.append(avalable_tokens)
                count_batch_token += avalable_tokens

            info.is_decoding.append(False)
            prefill_batch_count += 1
            info.prefill_batch = prefill_batch_count

            if prefill_batch_count == max_prefill_batch or count_batch_token >= max_batch_tokens:
                break

        for chunk_prompt in info.chunk_prompts:
            info.merge_prompt.extend(chunk_prompt)

        return info

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
