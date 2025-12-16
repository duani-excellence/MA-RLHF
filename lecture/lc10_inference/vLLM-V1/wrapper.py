from model import PageToyModel
from kvcache import PageKVCacheEngine
from scheduler import SchedulerInfo

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Optional, Any

torch.manual_seed(42)


class ModelWrapper:
    """封装模型的前向传播, Wrapper 可以用于扩展分布式训练等功能"""

    def __init__(self, model: PageToyModel, kv_cache_manager: PageKVCacheEngine):
        self.model = model

    def forward(self,
                input_ids,
                kv_cache,
                info: SchedulerInfo):

        logits, KV = self.model.forward(input_ids, kv_cache, info=info)

        return logits, KV

    def generate_next_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """从logits生成下一个token(贪婪采样)"""
        if len(logits) == 0:
            return torch.tensor([])
        return torch.argmax(logits, dim=-1)
