from block_table import BlockTable
from config import vLLMEngineConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Optional, Any

torch.manual_seed(42)


class PageKVCacheEngine:
    """分页式管理 KV 缓存"""

    def __init__(self, config: vLLMEngineConfig):
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
        self.kv_len = {}

    def get_kv_cache(self, request_ids: List[int] = []):

        if len(request_ids) == 0:
            return [], []

        num_pages_len = []
        KVs = []

        for _, idx in enumerate(request_ids):
            if not idx in self.request_to_pages:
                KV = None
                num_pages_len.append(0)
            else:
                page_ids = self.request_to_pages[idx]
                num_pages_len.append(len(page_ids))
                num_pages = len(page_ids)
                KV = self.kv_cache[:, :, page_ids, ...].clone()
                KVs.append(KV)

        # 这里的 kv cache 是 PD 无感的
        return KVs, num_pages_len

    def update_kv_cache(self, request_id, KV):
        """ 实现逻辑
        1. 检查 reqeust 对应的 Page-KV-Cache 的空余量
        2. 如果空余量为 0, 申请 1 个新的页
        3. 如果空余量不为 0, 将部分 KV 填入到 request 的最后一页
        """
        _, L, T, H, D = KV.shape
        if request_id not in self.kv_len:
            self.kv_len[request_id] = 0
            page_ids = self.block_table._allocate_pages(num_pages=1)
            self.request_to_pages[request_id] = page_ids
            self.page_to_request[page_ids[0]] = request_id
        else:
            page_ids = self.request_to_pages[request_id]

        new_len = T
        while new_len != 0:
            tmp_kv_len = self.kv_len[request_id]
            page_ids = self.request_to_pages[request_id]
            cur_total_len = self.page_size*len(page_ids)
            page_avaliable_len = cur_total_len-tmp_kv_len

            if page_avaliable_len == 0:
                # page-kv 没有空余的空间, 申请空间
                if len(page_ids) > 0:
                    parent_page_ids = page_ids[-1]
                else:
                    parent_page_ids = -1
                page_id = self.block_table._allocate_pages(
                    num_pages=1, parent_block_id=parent_page_ids)[0]
                # if page_id == -1: allocate faild
                page_avaliable_len = self.page_size

                self.request_to_pages[request_id].append(page_id)
                self.page_to_request[page_id] = request_id

            else:
                if page_avaliable_len >= new_len:  # 现有页面够保存新 kv
                    target_page_id = page_ids[-1]
                    target_pos_start = self.page_size - page_avaliable_len
                    target_pos_end = target_pos_start + new_len
                    self.kv_cache[:, :, target_page_id,
                                  target_pos_start:target_pos_end, :, :] = KV[:, :, T-new_len:]
                    new_len = 0
                else:  # 现有页面不足够保存新 KV
                    target_page_id = page_ids[-1]
                    target_pos_start = self.page_size - page_avaliable_len
                    # target_pos_end = target_pos_start + new_len

                    self.kv_cache[:, :, target_page_id, target_pos_start:, :,
                                  :] = KV[:, :, T-new_len: T-new_len+page_avaliable_len]

                    new_len -= page_avaliable_len
                self.kv_len[request_id] += page_avaliable_len
            if new_len == 0:
                break
        return self.kv_len[request_id]

    def free(self, request_id: int):
        """释放请求占用的页面"""
        allocate_pages_ids = self.request_to_pages[request_id]
        self.block_table._free_pages(allocate_pages_ids)

        self.kv_cache[:, :, allocate_pages_ids, :, :, :] = 0

        del self.request_to_pages[request_id]
        for idx in allocate_pages_ids:
            del self.page_to_request[idx]
        print(
            f"[FREE] request ID{request_id} pages, len{len(allocate_pages_ids)}")

    def get_request_info(self, ):

        for request_id, page_ids in self.request_to_pages.items():
            print(f'----Req.ID:{request_id}----')
            print('pages list:', page_ids)
            print('cur_length:', self.kv_len[request_id])
