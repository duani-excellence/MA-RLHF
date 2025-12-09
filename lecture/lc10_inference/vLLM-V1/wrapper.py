from .model import PageToyModel
from .kvcache import PageKVCacheEngine

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Optional, Any

torch.manual_seed(42)

class ModelWrapper:
    """封装模型的前向传播"""

    def __init__(self, model: PageToyModel, kv_cache_manager: PageKVCacheEngine):
        self.model = model
        self.cacher = kv_cache_manager

    def prefill_requests(self, request_ids: List[int], prompts: List[List[int]]):
        """
        预填充新请求
            1. 申请pages
            2. 组page级别的input_ids
        返回: request_page_ids:{[1,3,2], [5,4]}

        """
        if len(request_ids) == 0:
            return torch.tensor([])
        T = self.cacher.page_size

        # 数据预处理
        request_page_ids = []
        request_num_pages = []
        input_ids_list = []
        for request_id, prompt in zip(request_ids, prompts):

            # 获取 requst -> page_ids
            tmp_page_ids = self.cacher.allocate_request_pages(
                request_id,  len(prompt))
            tmp_len = len(tmp_page_ids)
            request_page_ids.append(tmp_page_ids)
            request_num_pages.append(tmp_len)

            # batch input_ids -> page input ids
            requst_input_ids = torch.zeros(
                len(tmp_page_ids), T, dtype=torch.long)  # padding tensor
            for i in range(tmp_len):
                if i == tmp_len-1:
                    offset = len(prompt) % T
                    requst_input_ids[i, :offset] = torch.tensor(
                        prompt[i*T:  len(prompt)], dtype=torch.long)
                else:
                    requst_input_ids[i, :] = torch.tensor(
                        prompt[i*T: (i+1)*T], dtype=torch.long)

                input_ids_list.append(requst_input_ids)
        input_ids = torch.cat(input_ids_list, dim=0)

        # 执行预填充
        with torch.no_grad():
            logits, layer_kvcaches = self.model.forward(input_ids,
                                                        request_num_pages=request_num_pages,)

        # page logits 上取最后一个块的数据
        _, _, vocab_size = logits.shape
        page_logits = torch.zeros(len(request_ids), vocab_size)
        offset = 0
        for i in range(len(request_ids)):

            batch_id = request_num_pages[i]-1
            idx = len(prompts[i]) % T
            page_logits[i] = logits[batch_id+offset, idx, :]
            offset += request_num_pages[i]

        return page_logits, layer_kvcaches, request_page_ids

    def decode_next_tokens(self,
                           next_tokens: torch.Tensor,
                           current_length=None,
                           num_pages_len=None,
                           KVCache=None) -> Tuple[torch.Tensor, Any]:
        """解码下一个token"""

        with torch.no_grad():
            logits, layer_kvcaches = self.model.forward(
                next_tokens,
                kvcaches=KVCache,
                request_num_pages=num_pages_len,
                current_length=current_length)

        return logits, layer_kvcaches

    def generate_next_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """从logits生成下一个token(贪婪采样)"""
        if len(logits) == 0:
            return torch.tensor([])
        return torch.argmax(logits, dim=-1)