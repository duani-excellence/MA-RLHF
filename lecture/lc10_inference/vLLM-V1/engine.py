import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Optional, Any

from .model import PageToyModel
from .kvcache import PageKVCacheEngine
from .schedular import Schedular
from .wrapper import ModelWrapper

torch.manual_seed(42)

class vLLMEngine:
    """vLLM 主引擎"""

    def __init__(self, model, config):
        self.cacher = PageKVCacheEngine(config)
        self.model_wrapper = ModelWrapper(model, self.cacher)
        self.schedular = Schedular(config.max_seq_len,)

    def add_request(self, prompt: List[int], max_seq_len) -> int:
        """添加新请求"""
        return self.schedular.add_request(prompt, max_seq_len)

    def step(self):
        """
        """
        request_ids = None
        layer_kvcaches = None
        # 阶段1: 处理解码(已有请求)
        if self.schedular.get_num_running_requests() > 0:

            request_ids = self.schedular.get_running_request_ids()

            # 准备输入token (上一个 step 生成的token)
            input_tokens = torch.tensor([
                self.schedular.requests[req_id].generated_tokens[-1]
                for req_id in request_ids
            ], dtype=torch.long)
            current_length = torch.tensor([
                self.schedular.requests[req_id].current_length
                for req_id in request_ids
            ], dtype=torch.long)
            input_tokens = input_tokens.unsqueeze(dim=1)

            # Page KVCache -> Batch KVCache
            # batch_kvcache = self.cacher.get_sequence_kvcache(request_ids)
            batch_kvcache, num_pages_len, batch_to_page = self.cacher.get_page_kvcache(
                request_ids)

            # 解码
            logits, layer_kvcaches = self.model_wrapper.decode_next_tokens(input_tokens,
                                                                           KVCache=batch_kvcache,
                                                                           num_pages_len=num_pages_len,
                                                                           current_length=current_length)
            next_tokens = self.model_wrapper.generate_next_tokens(logits)

            # update kv cache
            self.update_kvcache(request_ids, layer_kvcaches)

            # 更新状态
            for i, request_id in enumerate(request_ids):
                self.schedular.update_request(
                    request_id, next_tokens[i].item())
                if self.schedular.requests[request_id].is_finished():
                    self.cacher.free_request_pages(request_id)

        # 阶段2: 处理预填充(新请求)
        if self.schedular.get_num_pending_requests() > 0:
            pending_requests = self.schedular.get_pending_requests(
                config.num_pages
            )
            request_ids = [idx for idx, _ in pending_requests]
            prompts = [prompt for _, prompt in pending_requests]

            if pending_requests:
                logits, layer_kvcaches, request_page_ids = self.model_wrapper.prefill_requests(
                    request_ids, prompts)
                next_tokens = self.model_wrapper.generate_next_tokens(logits)
                for i, (request_id, _) in enumerate(pending_requests):
                    self.schedular.update_request(
                        request_id,
                        next_tokens[i].item()
                    )

                self.update_kvcache(
                    request_ids, layer_kvcaches, request_page_ids)

    def update_kvcache(self, request_ids, layer_kvcaches, request_page_ids=None):
        if request_ids != None:
            for i, idx in enumerate(request_ids):
                tmp_cache = [[layer_kvcache[0][i], layer_kvcache[1][i]]
                             for layer_kvcache in layer_kvcaches]
                self.cacher.update_pages(idx, tmp_cache)

    def has_pending_work(self) -> bool:
        """检查是否还有未完成的工作"""
        return self.schedular.has_pending_requests()

    def get_requests_info(self):
        pending = self.schedular.get_num_pending_requests()
        running = self.schedular.get_num_running_requests()
        total_request = len(self.schedular.requests)
        return pending, running, total_request