import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional, Any

from .kernel import page_attention_decoding_kernel, page_attention_prefill_kernel


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

    def forward_prefill(self,
                        X,
                        attention_mask=None,  # Attention mask 也是 page 型的
                        request_num_pages: List[int] = [],
                        request_length=None,
                        KV_Cache=[],
                        ):
        # Prefill
        B, T, _ = X.shape
        H = self.num_heads
        D = self.head_dim

        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
        Q = Q.reshape(B, T, H, D).transpose(1, 2)
        K = K.reshape(B, T, H, D).transpose(1, 2)
        V = V.reshape(B, T, H, D).transpose(1, 2)
        O = torch.zeros_like(Q)

        request_size = len(request_num_pages)
        offset = [0] * request_size
        for i in range(1, request_size):
            offset[i] = offset[i-1] + request_num_pages[i]

        for t in range(request_size):  # Request Loop
            offset_i = offset[t]
            N = request_num_pages[t]
            Q_ = Q[offset_i: offset_i+N]
            K_ = K[offset_i: offset_i+N]
            V_ = V[offset_i: offset_i+N]

            O_ = page_attention_prefill_kernel(Q_, K_, V_, mask=attention_mask)
            O[offset_i: offset_i+N] = O_

        O = O.transpose(1, 2).reshape(B, T, H*D)
        O = self.WO(O)
        O = X + self.act(O)

        return O, [K.transpose(1, 2), V.transpose(1, 2)]

    def forward_decoding(self,
                         X,
                         attention_mask=None,  # Attention mask 也是 page 型的
                         request_num_pages: List[int] = [],
                         request_length=None,
                         KV_Cache: List[torch.Tensor]=[],
                         ):
        # Decoding
        B, T, _ = X.shape
        H = self.num_heads
        D = self.head_dim

        # Proj
        q, k, v = self.WQ(X), self.WK(X), self.WV(X)
        q = q.reshape(B, 1, H, D).transpose(1, 2)
        k = k.reshape(B, 1, H, D).transpose(1, 2)
        v = v.reshape(B, 1, H, D).transpose(1, 2)

        # TODO: Apply RoPE For Q,K

        # step1: init
        S = q @ k.transpose(2, 3)  # B, H, 1, 1)
        M_ = S.clone()
        L_ = torch.ones_like(M_)
        O_ = v

        # step2: repeat q (dispatch)
        repeat_tensor = torch.tensor(request_num_pages)
        q_ = torch.repeat_interleave(q, repeat_tensor, dim=0)

        # step3: block attenion,
        K_, V_ = KV_Cache[0], KV_Cache[1]  # bsz, seq_len, num_head, head_dim
        S = q_ @ K_.transpose(1, 2).transpose(2, 3)
        # TODO: Mask
        M, _ = torch.max(S, dim=-1, keepdim=True)
        L = torch.sum(torch.exp(S - M), dim=-1, keepdim=True)
        P = torch.softmax(S, dim=-1)
        O = P @ V_.transpose(1, 2)

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
            break

        O = globle_O.transpose(1, 2).reshape(B, 1, H*D)
        O = self.WO(O)
        O = X + self.act(O)

        return O, [k.transpose(1, 2), v.transpose(1, 2)]


class PageToyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)
        self.decoder = nn.ModuleList(
            [PageAttentionBlock(config) for i in range(config.num_layers)]
        )

    def forward(self, x, kvcaches=None, current_length=None, request_num_pages=None):
        layer_kvcaches = []
        X = self.embd(x)

        for i, block in enumerate(self.decoder):
            if kvcaches == None:
                X, layer_kvcache = block.forward_prefill(
                    X, request_num_pages=request_num_pages)
            else:
                X, layer_kvcache = block.forward_decoding(X,
                                                          KV_Cache=[kvcaches[0][i],
                                                                    kvcaches[1][i]],
                                                          request_length=current_length,
                                                          request_num_pages=request_num_pages)

            layer_kvcaches.append(layer_kvcache)
        logits = self.lm_head(X)
        return logits, layer_kvcaches
