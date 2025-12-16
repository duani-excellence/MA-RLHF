import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Optional, Any

from model import PageToyModel
from kvcache import PageKVCacheEngine
from scheduler import Scheduler, SchedulerInfo
from wrapper import ModelWrapper

torch.manual_seed(42)


class vLLMEngine:
    """vLLM 主引擎"""

    def __init__(self, model, config):
        self.cacher = PageKVCacheEngine(config)
        self.model_wrapper = ModelWrapper(model, self.cacher)
        self.scheduler = Scheduler(config.max_seq_len,)
        self.num_pages = config.num_pages

    def add_request(self, prompt: List[int], max_seq_len) -> int:
        """添加新请求"""
        return self.scheduler.add_request(prompt, max_seq_len)

    def get_merge_batch(self, info: SchedulerInfo):
        # 1 x seq_len
        input_ids = torch.tensor([info.merge_prompt], dtype=torch.long)
        return input_ids

    def update(self, next_token, KV, info: SchedulerInfo):
        """
        next_token request-level,
        kv: merge-prompt split to update page_kvcache, shape: 2 x L x T x H x d
        """

        for bid, token in enumerate(next_token):
            req_id = info.ids[bid]
            if token != -1:
                # self.scheduler.requests[req_id].add_token(token)
                self.scheduler.update_request(req_id, token)

        reqs_KV = KV.split(info.chunk_len, dim=2)
        for req_id, tmp_KV in zip(info.ids, reqs_KV):
            if self.scheduler.requests[req_id].status == "REQUEST_COMPLETED":
                self.cacher.free(req_id)
            else:
                new_kv_len = self.cacher.update_kv_cache(req_id, tmp_KV)
                self.scheduler.requests[req_id].kv_len = new_kv_len

            # self.scheduler.update_request()

    def execute(self, input_ids: torch.Tensor,
                kv_cache: List[torch.Tensor],
                info: SchedulerInfo):
        logits, KV = self.model_wrapper.forward(input_ids,
                                                kv_cache,
                                                info)

        # 获取 next-token
        next_token = torch.argmax(logits, dim=-1)

        reqs_next_token = torch.split(next_token,
                                      info.chunk_len,
                                      dim=0)

        tokens = []
        for bid, token_ids in enumerate(reqs_next_token):
            token = token_ids[-1]
            if info.is_decoding[bid] == True:
                tokens.append(token)
            else:
                if info.last_pos[bid] != -1:
                    tokens.append(token)
                else:
                    tokens.append(-1)

        return tokens, KV

    def step(self):
        """
        step 函数, 采用 chunked-prefill 方式融合 P/D batch
        1. 获取融合batch
        2. 计算
        3. 生成 next token
        4. 更新状态/KVCache

        :param self: 说明
        """
        if self.scheduler.get_available_request() == 0:
            return

        # 获取 batch
        info = self.scheduler.get_requests()
        print(f'batch_info: P{info.prefill_batch}, D{info.decoding_batch}')
        kv_cache, info.kv_page_len = self.cacher.get_kv_cache(info.ids)
        input_ids, = self.get_merge_batch(info)

        # 执行计算
        next_token, kv = self.execute(input_ids, kv_cache, info)

        # 更新 requests 信息, 更新 KVCache
        self.update(next_token, kv, info)

        return

    def has_pending_work(self) -> bool:
        """检查是否还有未完成的工作"""
        return self.scheduler.has_pending_requests()

    def get_requests_info(self):
        pending = self.scheduler.get_num_pending_requests()
        running = self.scheduler.get_num_running_requests()
        total_request = len(self.scheduler.requests)
        return pending, running, total_request
