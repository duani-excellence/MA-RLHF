import torch
import torch.nn as nn
import torch.nn.functional as F

def page_attention_prefill_kernel(Q, K, V, mask=None):
    """
    1 Request(batch_size=1), [Flash Attention-V2](https://zhuanlan.zhihu.com/p/670085985)
    Args
        Q: num_pages, num_heads, seq_len, head_dim (in decoding, seq_len=1)
        K: num_pages, num_heads, seq_len, head_dim
        V: num_pages, num_heads, seq_len, head_dim
        mask: num_pages, seq_len, seq_len
    Output
        O: num_pages, num_heads, seq_len, head_dim
    """

    N, H, T, D = Q.shape  # batch_size, num_heads, seq_len, head_dim

    O_global = torch.zeros(N, H, T, D)
    for i in range(N):  # Q Loop
        O = torch.zeros(1, H, T, 1)
        M = torch.zeros(1, H, T, 1)
        L = torch.zeros(1, H, T, 1)
        Q_ = Q[i]
        for j in range(N):  # KV Loop

            if j > i:
                continue
            K_, V_ = K[j], V[j]

            S_ij = Q_ @ K_.transpose(1, 2)  # num_heads, seq_len, seq_len
            # num_heads, seq_len, 1
            M_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
            M_new = torch.maximum(M_ij, M)
            P_ij = torch.exp(S_ij - M_new)
            # num_heads, seq_len, 1
            L_ij = torch.sum(P_ij, dim=-1, keepdim=True)
            L_new = torch.exp(M - M_new) * L + L_ij
            O_i = torch.exp(M - M_new) * O + P_ij @ V_

            M = M_new
            L = L_new

        # re-scaled
        O_global[i] = (O_i / L_new).unsqueeze(dim=0)

    return O_global


def page_attention_decoding_kernel(O, M, L, O_, M_, L_):
    """
    online softmax trick
    """
    O = torch.cat([O, O_.unsqueeze(dim=0)], dim=0)
    M = torch.cat([M, M_.unsqueeze(dim=0)], dim=0)
    L = torch.cat([L, L_.unsqueeze(dim=0)], dim=0)

    M_new, _ = torch.max(M, dim=0, keepdim=True)
    L_new = torch.exp(M-M_new) * L
    L_new = torch.sum(L_new, dim=0, keepdim=True)
    O_new = (M-M_new) * (L/L_new) * O

    O_new = torch.sum(O_new, keepdim=True, dim=0)

    return O_new