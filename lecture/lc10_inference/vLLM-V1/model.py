import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional, Any

from kernel import page_attention_decoding_kernel, page_attention_prefill_kernel
from scheduler import SchedulerInfo


class PageAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.WQ = nn.Linear(config.dim, config.dim, bias=False)
        self.WK = nn.Linear(config.dim, config.dim, bias=False)
        self.WV = nn.Linear(config.dim, config.dim, bias=False)
        self.WO = nn.Linear(config.dim, config.dim, bias=False)
        self.act = nn.ReLU()

    def forward(self,
                X,
                attention_mask: torch.Tensor,  # Attention mask 也是 page 型的
                request_num_pages: List[int],
                request_length: List[int],
                KV_Cache: List[torch.Tensor],
                info: SchedulerInfo,
                ):

        N_p = info.prefill_batch
        N_d = info.decoding_batch
        H, D = self.num_heads, self.head_dim
        T, _ = X.shape

        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)

        Q = Q.reshape(T, H, D)
        K = K.reshape(T, H, D)
        V = V.reshape(T, H, D)
        return_KV = torch.stack((K, V), dim=0)

        # split to requst
        Q = Q.split(info.chunk_len, dim=0)
        K = K.split(info.chunk_len, dim=0)
        V = V.split(info.chunk_len, dim=0)

        if N_d > 0:
            Q_d, K_d, V_d = Q[:N_d], K[:N_d], V[:N_d]  # 1, T_i, H, D

            kv_page_len = info.kv_page_len[:N_d]

            KV_cache_d = torch.cat(KV_Cache[:N_d], dim=1)  # 2, N_D

            O_d = self.forward_decoding(Q_d, K_d, V_d,
                                        attention_mask,
                                        request_num_pages=kv_page_len,
                                        KV_Cache=KV_cache_d,
                                        request_length=[])

        if N_p > 0:
            Q_p, K_p, V_p = Q[N_d:], K[N_d:], V[N_d:]
            kv_page_len = info.kv_page_len[N_d:]

            O_p = self.forward_prefill(Q_p, K_p, V_p,
                                       attention_mask,
                                       request_num_pages=kv_page_len,
                                       KV_Cache=KV_Cache[:N_p],
                                       request_length=[])

        if N_p > 0 and N_d == 0:
            O = O_p
        elif N_d > 0 and N_p == 0:
            O = O_d
        elif N_d > 0 and N_p > 0:
            O = torch.cat((O_d, O_p), dim=0)
        else:
            O = None

        return O, return_KV

    def forward_prefill(self,
                        Q,
                        K,
                        V,
                        attention_mask: torch.Tensor,  # Attention mask 也是 page 型的
                        request_num_pages: List[int],
                        request_length: List[int],
                        KV_Cache: List[torch.Tensor],
                        ):
        # Prefill

        request_size = len(request_num_pages)
        offset = [0] * request_size
        for i in range(1, request_size):
            offset[i] = offset[i-1] + request_num_pages[i]

        O = []
        for t in range(request_size):  # Request Loop
            N = request_num_pages[t]

            # QKV 是 chunked-prompt 序列, 由于使用 page-kv-cache，cache 是 Page-level 列表。
            # prefill 内核实现有两种方法：
            # 1. 如果将序列QKV padding 成 Page-level。就变成标准的 Prefill-Page-Attention 形式计算。
            # 2. 将 KV 与 KVCache 拼接, 采用 Flash-Attention 核计算注意力
            # 3. 为了方便实现，QKV不切分, KV 与历史 KV-Cache 形成列表

            Q_ = Q[t].unsqueeze(dim=0)  # Q: 1, T, H, D
            if len(KV_Cache) == 0:
                K_ = K[t]
                V_ = V[t]
            elif len(KV_Cache) != 0 and len(KV_Cache[t]) == 0:
                K_ = K[t]
                V_ = V[t]
            else:
                K_ = [KV_Cache[t][0], K[t]]
                V_ = [KV_Cache[t][1], V[t]]

            # return

            # TODO: attention mask 要做成 chunked-prefill 式
            O_ = page_attention_prefill_kernel(Q_, K_, V_, attention_mask)

            O.append(O_.squeeze(dim=0))  # 1,T,H,D

        O = torch.cat(O, dim=0)
        T, H, D = O.shape
        O = O.reshape(T, H*D)
        O = self.WO(O)

        return O

    def forward_decoding(self,
                         q,
                         k,
                         v,
                         attention_mask: torch.Tensor,  # Attention mask 也是 page 型的
                         request_num_pages: List[int],
                         request_length: List[int],
                         KV_Cache: List[torch.Tensor],
                         ):
        # Decoding
        q = torch.cat(q, dim=0)
        k = torch.cat(k, dim=0)
        v = torch.cat(v, dim=0)

        B, H, D = q.shape

        # TODO: Apply RoPE For Q,K

        # step1: init
        # S = q @ k.transpose(2, 3)  # B, H, 1, 1)
        S = q[:, :, None, :] @ k[:, :, :, None]
        M_, _ = torch.max(S, dim=-1, keepdim=True)  # B, H, 1
        L_ = torch.ones_like(M_)
        O_ = v[:, :, None, :]

        # step2: repeat q (dispatch)
        repeat = torch.tensor(request_num_pages, dtype=torch.long)
        q_ = q.repeat_interleave(repeat, dim=0)

        # step3: block attenion,request-level KVCache
        Q_ = q_[:, :, None, :]  # num_page, h, 1, d
        K_, V_ = KV_Cache[0].transpose(1, 2), KV_Cache[1].transpose(
            1, 2)  # num_page, h, page_size, d
        Qshape = Q_.shape
        Kshape = K_.shape
        Vshape = V_.shape
        S = Q_ @ K_.transpose(2, 3)
        # TODO: Mask
        M, _ = torch.max(S, dim=-1, keepdim=True)
        L = torch.sum(torch.exp(S - M), dim=-1, keepdim=True)
        P = torch.softmax(S, dim=-1)
        O = P @ V_

        # step4: reudce result, (combine)
        offset = 0
        globle_O = torch.zeros_like(O_)
        for i, T in enumerate(request_num_pages):
            Oi = page_attention_decoding_kernel(
                O[offset:offset+T],
                M[offset:offset+T],
                L[offset:offset+T],
                O_[i],
                M_[i],
                L_[i],
            )
            globle_O[i] = Oi[0]  # BH1D
            offset += T

        O = globle_O.transpose(1, 2).reshape(B, H*D)

        return O


class PageToyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)
        self.decoder = nn.ModuleList(
            [PageAttentionBlock(config) for i in range(config.num_layers)]
        )

    def forward(self, x,
                kvcaches: List[torch.Tensor],
                current_length=None,
                request_num_pages=None,
                info=None):
        layer_kvcaches = []
        X = self.embd(x)

        x_embd_shape = x.shape
        X_embd_shape = X.shape

        for i, block in enumerate(self.decoder):

            tmp_kvcaches = []
            for kvcache in kvcaches:
                if kvcache is None:
                    tmp_kvcaches.append(None)
                else:
                    tmp_kvcaches.append(kvcache[:, i])  # request_level

            X_pre_shape = X.shape
            X, layer_kvcache = block.forward(X,
                                             KV_Cache=tmp_kvcaches,
                                             request_length=current_length,
                                             request_num_pages=request_num_pages,
                                             attention_mask=None,
                                             info=info)

            layer_kvcaches.append(layer_kvcache)
        logits = self.lm_head(X)
        return logits, torch.stack(layer_kvcaches, dim=1)
