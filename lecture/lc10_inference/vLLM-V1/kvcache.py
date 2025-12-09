from .block_table import BlockTable
from .config import vLLMEngineConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Optional, Any

torch.manual_seed(42)


class PageKVCacheEngine:
    """分页式管理 KV 缓存"""

    def __init__(self, config):
        # 初始化KV缓存 [layer, batch, seq, head, dim]

        self.num_pages = config.num_pages
        self.page_size = config.page_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads

        self.block_table = BlockTable(self.page_size,
                                      self.num_pages,)

        # KV 的存储表由块表大小来管理
        self.k_cache = torch.zeros(config.num_layers,
                                   self.num_pages,
                                   self.page_size,
                                   config.num_heads,
                                   config.head_dim)
        self.v_cache = torch.zeros_like(self.k_cache)

        # 每个请求的长度
        self.sequence_lengths = {}

        # 请求与 block_id 的映射信息
        self.request_to_pages = {}  # request_id -> [page_id1, page_id2, ...]
        self.page_to_request = {}  # page_id -> request_id

    def has_active_requests(self) -> bool:
        """检查是否有活跃的请求"""
        return len(self.request_to_pages) > 0

    def has_available_pages(self, request_length) -> bool:
        """检查是否有可用的分页"""
        free_num_pages = self.block_table.get_free_count()
        return request_length < free_num_pages * self.page_size

    def allocate_request_pages(self, request_id, request_length) -> List[int]:
        """
        为 prefill 请求预分配页面, 对于正在解码的 decoding 请求，不需要预分配 page, 后续分配功能写在一起
        """

        allocate_pages_size = (request_length // self.page_size)+1
        allocate_pages_ids = self.block_table._allocate_pages(
            allocate_pages_size)
        if allocate_pages_ids == None:
            print(f'[ALLOCATE] request ID{request_id} pages faild')
            return []

        self.request_to_pages[request_id] = allocate_pages_ids

        for i in self.request_to_pages[request_id]:
            self.page_to_request[i] = request_id
        self.sequence_lengths[request_id] = 0

        print(
            f'[ALLOCATE] request ID{request_id} pages len {len(allocate_pages_ids)}')

        return allocate_pages_ids

    def free_request_pages(self, request_id: int):
        """释放请求占用的页面"""
        allocate_pages_ids = self.request_to_pages[request_id]
        self.block_table._free_pages([0, 1])

        self.k_cache[:, allocate_pages_ids, :, :, :] = 0
        self.v_cache[:, allocate_pages_ids, :, :, :] = 0

        del self.request_to_pages[request_id]
        for idx in allocate_pages_ids:
            del self.page_to_request[idx]
        print(
            f"[FREE] request ID{request_id} pages, len{len(allocate_pages_ids)}")

    def update_pages(self, request_id, new_kv_cache):
        # 1. Prefill: 填充到 requst_id -> pages 上
        # 2. Decoding: 找到 requst_id -> pages 上的最后一个 token, 如果最后一个块已满，需要重新申请一个新的块表。

        # 1. Update Decoding blocks
        # 对于 Decoding 存在增加一个新 token 导致要新加一个 block 的情况
        seq_len, _, _ = new_kv_cache[0][0].shape  # 0 层数据 K数据
        T = self.page_size

        if seq_len == 1:
            length = self.sequence_lengths[request_id]
            pages_ids = self.request_to_pages[request_id]
            if length % T == 0:
                new_block_id = self.block_table._allocate_pages(
                    1, pages_ids[-1])

                self.request_to_pages[request_id].append(new_block_id[0])
                self.page_to_request[new_block_id[0]] = request_id
            self.sequence_lengths[request_id] += 1
            cur_offset_len = self.sequence_lengths[request_id] % T
        else:
            # prefill 填充 context 长度
            self.sequence_lengths[request_id] = seq_len

        # 2. Update Decoding Stage KV-Cache
        if seq_len == 1:
            for i, layer_kv_cache in enumerate(new_kv_cache):
                # seq_len, num_heads, head_dim = layer_kv_cache[0].shape
                pages_ids = self.request_to_pages[request_id]
                cur_offset_len = self.sequence_lengths[request_id] % T

                # 最后一个块上加 Cache
                self.k_cache[i, pages_ids[-1], cur_offset_len,
                             :, :] = layer_kv_cache[0][0, :, :]
                self.v_cache[i, pages_ids[-1], cur_offset_len,
                             :, :] = layer_kv_cache[1][0, :, :]

        # 3. Update Prefill Stage KV-Cache
        # 更新简单
        else:
            for i, layer_kv_cache in enumerate(new_kv_cache):
                pages_ids = self.request_to_pages[request_id]
                for k, idx in enumerate(pages_ids):
                    self.k_cache[i, pages_ids, :, :, :] = layer_kv_cache[0]
                    self.v_cache[i, pages_ids, :, :, :] = layer_kv_cache[1]

    def get_sequence_kvcache(self, request_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_page_kvcache(self, request_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 page2batch 数据, request-wise"""
        L, _, T, H, D = self.k_cache.shape

        N = 0  # num_pages
        batch_to_page = {}
        num_pages_len = []
        for t, idx in enumerate(request_ids):
            page_ids = self.request_to_pages[idx]
            num_pages_len.append(len(page_ids))
            for i, page_id in enumerate(page_ids):
                batch_to_page[i+N] = page_id
            N += len(page_ids)

        K = torch.zeros(L, N, T, H, D)
        V = torch.zeros(L, N, T, H, D)

        b_id = 0  # batch id start
        for _, idx in enumerate(request_ids):
            page_ids = self.request_to_pages[idx]
            num_pages = len(page_ids)

            K[:, b_id: b_id+num_pages, :, :, :] = self.k_cache[:, page_ids, :, :, :]
            V[:, b_id: b_id+num_pages, :, :, :] = self.v_cache[:, page_ids, :, :, :]

            b_id += num_pages

        return (K, V), num_pages_len, batch_to_page

    def get_request_info(self, ):

        for request_id, page_ids in self.request_to_pages.items():
            print(f'----Req.ID:{request_id}----')
            print('pages list:', page_ids)
            print('cur_length:', self.sequence_lengths[request_id])
