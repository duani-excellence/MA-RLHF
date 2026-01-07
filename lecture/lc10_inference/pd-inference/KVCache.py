# PD 分离有 3 种 KVCache 引擎
#  1. KVCache 仅存于 Decoding 节点
#  2. KVCache 在 Prefill / Decoding 各存各的, Prefill 节点 Cache 用于 Chunked-Prefill
#  3. KVCache 中心化/去中心化服务, Prefill节点和 Decoding 节点都可以从 Cache 服务中存取数据
# 本示例采用 1 便于实现核心 PD 分离

import torch
import ray


@ray.remote
class DistributedKVCacheEngine:
    def __init__(self, config):
        self.kv_cache_batch = config.kv_cache_batch
        self.kv_cache_len = config.kv_cache_len
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads

        # 假设预留的 KVCache 是足量的, 无须 page 化
        self.kv_cache = torch.zeros(2,
                                    config.num_layers,
                                    config.kv_cache_batch,
                                    config.kv_cache_len,
                                    config.num_heads,
                                    config.head_dim,)

        # 请求与 block_id 的映射信息
        self.request_to_batch = {}  # request_id -> [ kv_batch_id]
        self.batch_to_request = {}  # kv_batch_id -> request_id
        # self.kv_len = {}

    def get_kv_cache(self, reqs):
        if len(reqs) == 0:
            return None
        kv_batch_ids = []
        for req_id in reqs:
            kv_batch_ids = self.request_to_batch[req_id]
        kv_cache = self.kv_cache[:, :, kv_batch_ids]
        return kv_cache

    async def update_from_prefill(self, reqs, kv):
        start_ids = len(self.request_to_batch)
        len_ids = len(reqs)
        for i in range(len_ids):
            self.request_to_batch[reqs[i]] = start_ids + i
            self.batch_to_request[reqs[i]] = i + start_ids

        self.kv_cache[:, :, start_ids: start_ids+len_ids] = kv

    def update_from_decoding(self, reqs, kv, decoding_idx):
        # len_ids = len(reqs)
        for i, req_id in enumerate(reqs):
            pos = decoding_idx[i]
            batch_id = self.request_to_batch[req_id]
            self.kv_cache[:, :, batch_id, pos] = kv[:, :, i, 0] # 2,3,1,1,2,8

    def free(self, reqs):
        for req_id in reqs:
            batch_id = self.request_to_batch[req_id]
            self.kv_cache[:, : batch_id, :] = 0
