from .block_table import BlockTable
from .config import vLLMEngineConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Optional, Any

torch.manual_seed(42)


class PageKVCacheEngine:
    """分页式管理 KV 缓存"""

    def __init__(self, config : vLLMEngineConfig ):
        # 初始化KV缓存 [layer, batch, seq, head, dim]

        self.num_pages = config.num_pages
        self.page_size = config.page_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads

        self.block_table = BlockTable(self.page_size,
                                      self.num_pages,)

        # KV 的存储表由块表大小来管理
        self.kv_cache = torch.zeros(2,
                                   config.num_layers,
                                   self.num_pages,
                                   self.page_size,
                                   config.num_heads,
                                   config.head_dim,)

        # 每个请求的长度

        # 请求与 block_id 的映射信息
        self.request_to_pages = {}  # request_id -> [page_id1, page_id2, ...]
        self.page_to_request = {}  # page_id -> request_id
    
    def get_kv_cache(self, request_ids: List[int] = []):

        if len(request_ids) == 0:
            return [], []
        
        num_pages_len = []
        KVs = []
        
        for _, idx in enumerate(request_ids):
            page_ids = self.request_to_pages[idx]
            num_pages_len.append(len(page_ids))
            num_pages = len(page_ids)
            if num_pages == 0:
                KV = None
            else:
                KV = self.kv_cache[:, :, page_ids, :, :, :].clone()
                KVs.append(KV)

        # 这里的 kv cache 是 PD 无感的
        return KVs, num_pages_len
    
    def update_kv_cache(self, idx, KV):
        _, L, T, H, D= KV.shape

    def free(self, idx):
        return


    
    
    # def allocate_request_pages(self, request_id, request_prompt):

    # def has_active_requests(self) -> bool:
    #     """检查是否有活跃的请求"""
    #     return len(self.request_to_pages) > 0

    # def has_available_pages(self, request_length) -> bool:
    #     """检查是否有可用的分页"""
    #     free_num_pages = self.block_table.get_free_count()
    #     return request_length < free_num_pages * self.page_size

    # def allocate_request_pages(self, request_id, request_length) -> List[int]:
    #     """
    #     为 prefill 请求预分配页面, 对于正在解码的 decoding 请求，不需要预分配 page, 后续分配功能写在一起
    #     """

    #     allocate_pages_size = (request_length // self.page_size)+1
    #     allocate_pages_ids = self.block_table._allocate_pages(
    #         allocate_pages_size)
    #     if allocate_pages_ids == None:
    #         print(f'[ALLOCATE] request ID{request_id} pages faild')
    #         return []

    #     self.request_to_pages[request_id] = allocate_pages_ids

    #     for i in self.request_to_pages[request_id]:
    #         self.page_to_request[i] = request_id
    #     self.sequence_lengths[request_id] = 0

    #     print(
    #         f'[ALLOCATE] request ID{request_id} pages len {len(allocate_pages_ids)}')

    #     return allocate_pages_ids

    # def free_request_pages(self, request_id: int):
    #     """释放请求占用的页面"""
    #     allocate_pages_ids = self.request_to_pages[request_id]
    #     self.block_table._free_pages([0, 1])

    #     self.k_cache[:, allocate_pages_ids, :, :, :] = 0
    #     self.v_cache[:, allocate_pages_ids, :, :, :] = 0

    #     del self.request_to_pages[request_id]
    #     for idx in allocate_pages_ids:
    #         del self.page_to_request[idx]
    #     print(
    #         f"[FREE] request ID{request_id} pages, len{len(allocate_pages_ids)}")

    # def update_pages(self, request_id, new_kv_cache):
    #     pass

    # def get_sequence_kvcache(self, request_ids: List[int]) :
    #     pass

    # def get_page_kvcache(self, request_ids: List[int]):
    #     """获取 page2batch 数据, request-wise"""
    #     L, _, T, H, D = self.k_cache.shape

    #     N = 0  # num_pages
    #     batch_to_page = {}
    #     num_pages_len = []
    #     for t, idx in enumerate(request_ids):
    #         page_ids = self.request_to_pages[idx]
    #         num_pages_len.append(len(page_ids))
    #         for i, page_id in enumerate(page_ids):
    #             batch_to_page[i+N] = page_id
    #         N += len(page_ids)

    #     K = torch.zeros(L, N, T, H, D)
    #     V = torch.zeros(L, N, T, H, D)

    #     b_id = 0  # batch id start
    #     for _, idx in enumerate(request_ids):
    #         page_ids = self.request_to_pages[idx]
    #         num_pages = len(page_ids)

    #         K[:, b_id: b_id+num_pages, :, :, :] = self.k_cache[:, page_ids, :, :, :]
    #         V[:, b_id: b_id+num_pages, :, :, :] = self.v_cache[:, page_ids, :, :, :]

    #         b_id += num_pages

    #     return (K, V), num_pages_len, batch_to_page

    # def get_request_info(self, ):

    #     for request_id, page_ids in self.request_to_pages.items():
    #         print(f'----Req.ID:{request_id}----')
    #         print('pages list:', page_ids)
    #         print('cur_length:', self.sequence_lengths[request_id])
