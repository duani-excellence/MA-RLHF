import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import TokenizerBase
from model import Transformer
from config import TransformerModelConfig
from utils import PAD_TOKEN, SOS_TOKEN

torch.manual_seed(42)


def encode_batch(data, tokenizer):
    N = len(data)
    _, input_list = tokenizer.encode(data)
    max_len = max(len(input_ids) for input_ids in input_list)

    input_ids = torch.ones(N, max_len, dtype=torch.long) * \
        tokenizer.vocab[PAD_TOKEN]
    for i, input_id in enumerate(input_list):
        input_ids[i, -len(input_id):] = torch.tensor(input_id,
                                                     dtype=torch.long)
    return input_ids


def generate(model,
             trg_tokenizer,
             inputs,
             max_new_tokens=100):
    src_mask = None
    X_src = None
    N, seq_len = inputs.shape
    Y = torch.ones(N, 1, dtype=torch.long) * trg_tokenizer.vocab[SOS_TOKEN]
    for i in range(max_new_tokens):
        # Inference Stage, only one-encoder forward, and multi-step decode
        logits, prob, src_mask, X_src = model(inputs, Y, src_mask, X_src)
        # logits = logits[:, -1, :]
        # prob = F.softmax(logits, dim=-1)
        prob = prob[:, -1, :]
        next_token_ids = torch.argmax(
            prob, dim=-1, keepdim=True)  # greed search
        Y = torch.cat((Y, next_token_ids), dim=-1)
    return Y


if __name__ == "__main__":

    # load tokenizer
    src_tokenizer = TokenizerBase()
    src_tokenizer.from_pretrained('./output/transformer/tokenizer_zh')
    trg_tokenizer = TokenizerBase()
    trg_tokenizer.from_pretrained('./output/transformer/tokenizer_en')

    # create model
    model, _, _ = Transformer.from_pretrained('./output/transformer/')

    data = ['小冬瓜学大模型。',
            '注意力机制',
            '大语言模型能提高效率。']

    inputs = encode_batch(data, src_tokenizer)
    # print(inputs.shape)
    outputs = generate(model, trg_tokenizer, inputs, max_new_tokens=100)
    outputs = outputs.tolist()

    output_str = trg_tokenizer.decode(outputs)
    for src, trg in zip(data, output_str):
        print('-'*100)
        print(src)
        print(''.join(trg))
