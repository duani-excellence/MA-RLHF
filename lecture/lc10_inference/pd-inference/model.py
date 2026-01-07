import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
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

    def forward(self, X, kvcache=None, current_length=None):
        bsz, seq_len, _ = X.shape
        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
        Q = Q.reshape(bsz, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2)
        K = K.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(bsz, seq_len, self.num_heads, self.head_dim)

        if kvcache is None:
            K_, V_ = K, V
        else:
            # Note: Ignore KV-cat
            K_ = kvcache[0]
            V_ = kvcache[1]
            if len(K_.shape) == 3:
                K_ = K_.unsqueeze(dim=0)
                V_ = V_.unsqueeze(dim=0)
            

        K_ = K_.transpose(1, 2)
        V_ = V_.transpose(1, 2)

        S = Q @ K_.transpose(2, 3)//math.sqrt(self.head_dim)
        P = F.softmax(S, dim=-1)
        Z = P@V_
        Z = Z.transpose(1, 2).reshape(bsz, seq_len, self.dim)
        O = self.WO(Z)

        # activate & shorcut
        O_ = X + self.act(O)

        return O_, torch.stack( [K, V], dim = 0)


class ToyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)
        self.decoder = nn.ModuleList(
            [DecoderBlock(config) for i in range(config.num_layers)]
        )

    def forward(self, x, kvcaches=None, current_length=None):
        layer_kvcaches = []
        X = self.embd(x)

        for i, block in enumerate(self.decoder):
            if kvcaches == None:
                X, layer_kvcache = block(X, None, None)
            else:
                # kvcaches[0][i]-> bsz, seq_len, heads, dim
                X, layer_kvcache = block(X,
                                         kvcache=[kvcaches[0][i], 
                                                  kvcaches[1][i]],
                                         current_length=current_length)
            layer_kvcaches.append(layer_kvcache)
        logits = self.lm_head(X)
        return logits, torch.stack(layer_kvcaches, dim = 1)
