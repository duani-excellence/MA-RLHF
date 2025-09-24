from config import TransformerModelConfig
from utils import get_src_mask, get_trg_mask, get_src_trg_mask
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)


class TransformerInputLayer(nn.Module):
    """
    词向量 + 位置编码
    """

    def __init__(self, vocab_size=100, dim=512, max_len=1024, base=10000.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.max_len = max_len

        theta_ids = torch.arange(0, dim, 2)  # 0, 2, 4, ..., 512
        theta = 1 / (base ** (theta_ids / dim))
        pe = torch.zeros(dim)  # 512, sin( theta_0 ),cos( theta_0), ...
        pe[theta_ids] = theta
        pe[theta_ids+1] = theta

        position_ids = torch.arange(0, max_len)  # 0, 1, 2, ..., 1024
        self.PE = torch.outer(position_ids, pe)  # 1024 x 512

        self.PE[:, theta_ids] = torch.sin(self.PE[:, theta_ids])
        self.PE[:, theta_ids+1] = torch.sin(self.PE[:, theta_ids+1])

    def forward(self, input_ids):
        """
        嵌入向量 + 绝对位置编码(标准实现)
        """
        bs, seq_len = input_ids.shape
        X = self.embedding(input_ids)
        PE = self.PE[:seq_len, :]
        X_ = X + PE
        return X_


class LayerNorm(nn.Module):
    def __init__(self, dim, ):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.epsilon = 1e-8

    def forward(self, X, ):
        mu = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True)
        X_hat = (X - mu) / torch.sqrt(var + self.epsilon)
        Y = X_hat * self.gamma + self.beta
        return Y


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, ):
        super().__init__()
        self.dim = dim
        self.W_up = nn.Linear(self.dim, 4 * self.dim)
        self.ReLU = nn.ReLU()
        self.W_down = nn.Linear(4 * self.dim, self.dim)

    def forward(self, X):
        X_ = self.ReLU(self.W_up(X))
        Y = self.W_down(X_)
        return Y


class MultiHeadScaleDotProductAttention(nn.Module):
    def __init__(self, dim_in, dim_out, heads=8):
        super().__init__()
        self.WQ = nn.Linear(dim_in, dim_out)
        self.WK = nn.Linear(dim_in, dim_out)
        self.WV = nn.Linear(dim_in, dim_out)
        self.WO = nn.Linear(dim_in, dim_out)
        self.heads = heads
        self.head_dim = dim_out // self.heads

    def forward(self, X_Q, X_K, X_V, mask=None):
        bs, seq_len, dim = X_Q.shape
        bs, seq_K_len, dim = X_K.shape
        bs, seq_V_len, dim = X_V.shape
        Q = self.WQ(X_Q)
        K = self.WK(X_K)
        V = self.WV(X_V)

        # 拆分维度
        Q_h = Q.view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K_h = K.view(bs, seq_K_len, self.heads, self.head_dim).transpose(
            1, 2)  # KV len 可以不等同于 Q len
        V_h = V.view(bs, seq_V_len, self.heads, self.head_dim).transpose(1, 2)

        # 多个 q_i 计算注意力特征
        # 1. 为什么要除于 \sqrt{d}
        S = Q_h @ K_h.transpose(2, 3) / math.sqrt(self.head_dim)

        if mask is not None:
            idx = torch.where(mask == 0)
            S[idx[0], :, idx[1], idx[2]] = -10000.0

        P = torch.softmax(S, dim=-1)  # 行 softmax
        Z = P @ V_h

        # 恢复维度
        Z = Z.transpose(1, 2).reshape(bs, seq_len, dim)

        output = self.WO(Z)

        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.attn = MultiHeadScaleDotProductAttention(dim, dim, heads)
        self.ln1 = LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim)
        self.ln2 = LayerNorm(dim)

    def forward(self, X, src_mask=None):
        X_attn = self.attn(X, X, X, mask = src_mask)
        X_ln = self.ln1(X_attn)
        X = X + X_ln

        X_ffn = self.ffn(X)
        X_ln = self.ln2(X_ffn)
        X = X + X_ln

        return X


class TransformerEncoder(nn.Module):
    """
    输入 原文本序列，输出 token 序列的编码表征
    输入:[bs, src_seq_len, dim]
    输出:[bs, src_seq_len, dim]
    """

    def __init__(self, dim=512, num_layers=6, heads=8):
        super().__init__()
        # self.encoder = nn.Linear(dim, dim)
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(dim, heads) for _ in range(num_layers)]
        )

    def forward(self, X, mask=None):
        for encode_block in self.encoder:
            X = encode_block(X, mask)
        return X


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.masked_attn = MultiHeadScaleDotProductAttention(dim, dim, heads)
        self.ln1 = LayerNorm(dim)

        self.cross_attn = MultiHeadScaleDotProductAttention(dim, dim, heads)
        self.ln2 = LayerNorm(dim)

        self.ffn = FeedForwardNetwork(dim)
        self.ln3 = LayerNorm(dim)

    def forward(self, X, X_src, trg_mask=None, src_trg_mask=None):
        X_attn = self.masked_attn(X, X, X, trg_mask)
        X_ln = self.ln1(X_attn)
        X = X + X_ln

        X_attn = self.cross_attn(X, X_src, X_src, src_trg_mask)
        X_ln = self.ln2(X_attn)
        X = X + X_ln

        X_ffn = self.ffn(X)
        X_ln = self.ln3(X_ffn)
        X = X + X_ln
        return X


class TransformerDecoder(nn.Module):
    """
    输入:[bs, trg_seq_len, dim]
    输出:[bs, trc_seq_len, dim]
    """

    def __init__(self, dim=512, num_layers=6, heads=8):
        super().__init__()
        # self.decoder = nn.Linear(dim, dim)
        self.decoder = nn.ModuleList(
            [TransformerDecoderBlock(dim, heads) for i in range(num_layers)]
        )

    def forward(self, X, X_src, trg_mask=None, src_trg_mask=None):
        for decoder_block in self.decoder:
            X = decoder_block(X, X_src, trg_mask=trg_mask,
                              src_trg_mask=src_trg_mask)
        return X


class TransformerOutputLayer(nn.Module):
    """
    """

    def __init__(self, vocab_size=100, dim=512):
        super().__init__()
        self.lm_head = nn.Linear(dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        logits = self.lm_head(X)
        prob = self.softmax(logits)
        return logits


class Transformer(nn.Module):
    """
    实际代码实现可以用 config 来初始化或传参, 提升代码简洁性。
    手动传参好处在于: 能够明白每层的具体的超参数、输入和输出。
    """

    def __init__(self, config: TransformerModelConfig = None):
        super().__init__()
        self.config = config

        self.encoder_input = TransformerInputLayer(vocab_size=self.config.src_vocab_size,
                                                   dim=self.config.dim,
                                                   max_len=self.config.max_len, )
        self.encoder = TransformerEncoder(dim=self.config.dim,
                                          num_layers=self.config.num_layers,
                                          heads=self.config.heads)
        self.decoder_input = TransformerInputLayer(vocab_size=self.config.trg_vocab_size,
                                                   dim=self.config.dim,
                                                   max_len=self.config.max_len, )
        self.decoder = TransformerDecoder(dim=self.config.dim,
                                          num_layers=self.config.num_layers,
                                          heads=self.config.heads)
        self.output_layer = TransformerOutputLayer(vocab_size=self.config.trg_vocab_size,
                                                   dim=self.config.dim)

    def forward(self, src_ids, trg_ids, src_mask=None, X_src=None):
        """
        输入:[bs, src_seq_len], [bs, trg_seq_len -1]
        输出:[bs, trg_seq_len - 1, trg_vocab_size]
        """

        # Prefill
        # 在 Inference 阶段，单次编码，多次解码
        if src_mask is None and X_src is None:
            src_mask = get_src_mask(
                src_ids, pad_token_id=self.config.src_pad_token_id)
            X = self.encoder_input(src_ids)
            X_src = self.encoder(X, src_mask)

        # Decoding
        trg_mask = get_trg_mask(
            trg_ids, pad_token_id=self.config.trg_pad_token_id)
        src_trg_mask = get_src_trg_mask(src_ids, trg_ids,
                                        src_pad_token_id=self.config.src_pad_token_id,
                                        trg_pad_token_id=self.config.trg_pad_token_id,)
        Y = self.decoder_input(trg_ids)
        Y = self.decoder(Y, X_src, trg_mask=trg_mask,
                         src_trg_mask=src_trg_mask)

        logits = self.output_layer(Y)
        prob = F.softmax(logits, dim=-1)

        return logits, prob, src_mask, X_src

    def save_pretrained(self, file_dir, optimizer=None):
        """保存模型和配置"""
        save_data = {
            'model_state_dict': self.state_dict(),
            # 'config': self.config,
            # 'config_dict': asdict(self.config)  # 用于JSON序列化
        }

        if optimizer:
            save_data['optimizer_state_dict'] = optimizer.state_dict()

        file_dir = Path(file_dir)
        if not file_dir.exists():
            file_dir.mkdir()

        model_path = file_dir / 'model.pth'
        config_path = file_dir / 'config.json'

        torch.save(save_data, model_path)
        self.config.save_json(config_path)

    @classmethod
    def from_pretrained(cls, file_dir, device='cpu'):
        """加载模型和配置, 可先加载至 CPU 再搬运到 GPU 设备"""

        file_dir = Path(file_dir)
        model_path = file_dir / 'model.pth'
        config_path = file_dir / 'config.json'

        data = torch.load(model_path, map_location=device, weights_only=False)

        config = TransformerModelConfig.from_json(config_path)
        model = cls(config)
        model.load_state_dict(data['model_state_dict'])
        model.to(device)

        optimizer_state = None
        if 'optimizer_state_dict' in data:
            optimizer_state = data.get('optimizer_state_dict')

        return model, config, optimizer_state


if __name__ == "__main__":
    # create model
    config = TransformerModelConfig(
        src_vocab_size=100,
        trg_vocab_size=200,
        max_len=512,
        dim=16,
        heads=8,
        num_layers=1,
        position_encoding_base=10000.0,
        src_pad_token_id=0,
        trg_pad_token_id=0,
    )

    # compute model
    model = Transformer(config)
    X = torch.randint(0, config.src_vocab_size, (2, 3), dtype=torch.long)
    Y = torch.randint(0, config.trg_vocab_size, (2, 7), dtype=torch.long)
    logits, _ = model(X, Y)
    print(logits.shape)
    print(logits[0, 0, :10])

    # save model
    model.save(file_dir='./output/test_model_io')
    del model

    # load model
    new_model, config, _ = Transformer.load(file_dir='./output/test_model_io')
    print(config)
    print(new_model)

    # check model
    logits, _ = new_model(X, Y)
    print(logits[0, 0, :10])
