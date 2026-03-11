import torch
from typing import Union, List


def page_attention_prefill_kernel(Q,
                                  K: Union[torch.Tensor, List[torch.Tensor]],
                                  V: Union[torch.Tensor, List[torch.Tensor]],
                                  mask=None):
    """
    Q: 1, T, H, D
    """

    Nr, T, H, D = Q.shape  # batch_size, num_heads, seq_len, head_dim

    if isinstance(K, torch.Tensor):
        K_cache = [K]
        V_cache = [V]
    else:
        """
        K[0]: num_pages, page_size, H, D
        K[1]: prompt_len, H, D
        """
        K_cache = list(K[0])
        K_cache.append(K[1])
        V_cache = list(V[0])
        V_cache.append(V[1])

    Nc = len(K_cache)
    O_global = torch.zeros(Nr, H, T, D)
    for i in range(Nr):  # Q Loop
        O = torch.zeros(H, T, D)
        M = torch.ones(H, T, 1) * -100000.0
        L = torch.zeros(H, T, 1)
        Q_ = Q[i].transpose(0, 1)

        for j in range(Nc):  # KV Loop
            K_ = K_cache[j].transpose(0, 1)
            V_ = V_cache[j].transpose(0, 1)  # 1, NH, T, DIM

            S_ij = Q_ @ K_.transpose(1, 2)  # num_heads, seq_len, seq_len
            # num_heads, seq_len, 1
            M_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)  # 4,8,1
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

    O_global = O_global.transpose(1, 2)  # 1, T, H, D
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


if __name__ == '__main__':

    # test 1: prefill
    # request-level  # num_pages, seq_len, num_heads, head_dim
    Q = torch.randn(1, 2, 3, 4)
    K = torch.randn(2, 3, 4)
    V = torch.randn(2, 3, 4)
    O = page_attention_prefill_kernel(Q, K, V)
    print(O.shape)

    # test 2: chunked-prefill
    Q = torch.randn(1, 2, 3, 4)  # request-level
    # Cache num_pages, page_size, H, D
    K = [torch.randn(5, 10, 3, 4), torch.randn(2, 3, 4)]
    V = [torch.randn(5, 10, 3, 4), torch.randn(2, 3, 4)]
    O = page_attention_prefill_kernel(Q, K, V)
    print(O.shape)
