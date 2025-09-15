import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
torch.manual_seed(42)

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
IGNORE_INDEX = -100


def token_pre_process(token_ids_list,
                      sos_token_id=None,
                      eos_token_id=None):
    token_ids_pre_process = []
    for token_ids in token_ids_list:
        if sos_token_id is not None:
            token_ids = [sos_token_id] + token_ids
        if eos_token_id is not None:
            token_ids = token_ids + [eos_token_id]
        token_ids_pre_process.append(token_ids)
    return token_ids_pre_process

# def padding_max_length(input_ids,
#                        max_len = 32,
#                        pad_token_id = None,
#                        padding_side = 'RIGHT',
#                       truction_side = 'RIGHT'):
#     if pad_token_id is None:
#         return
#     tokens_lens = [ len(ids) for ids in input_ids]
#     tokens_lens = torch.tensor(tokens_lens, dtype = torch.long)
#     tokens_max_len = torch.max(tokens_lens)

#     # trunction
#     if tokens_max_len > max_len:
#         tokens_max_len = max_len
#     if truction_side == 'RIGHT':
#         input_ids = [ ids[ : min(len(ids), tokens_max_len)] for ids in input_ids]
#     else:
#         input_ids = [ ids[ -min(len(ids), tokens_max_len) : ] for ids in input_ids]

#     # padding
#     paddding_input_ids = torch.ones(len(input_ids), tokens_max_len, dtype = torch.long) * pad_token_id
#     if padding_side == 'RIGHT':
#         for i in range(len(input_ids)):
#             paddding_input_ids[i, : len(input_ids[i])] = torch.tensor(input_ids[i], dtype = torch.long)
#     else: # left padding
#         for i in range(len(input_ids)):
#             paddding_input_ids[i, -len(input_ids[i]):] = torch.tensor(input_ids[i], dtype = torch.long)

#     return paddding_input_ids


def get_src_mask(input_ids, pad_token_id=0):
    bs, seq_len = input_ids.shape
    mask = torch.ones(bs, seq_len, seq_len)
    for i in range(bs):
        pad_idx = torch.where(input_ids[i, :] == pad_token_id)[0]
        mask[i, pad_idx, :] = 0
        mask[i, :, pad_idx] = 0
    return mask


def get_trg_mask(input_ids, pad_token_id=0):
    bs, seq_len = input_ids.shape
    mask = torch.tril(torch.ones(bs, seq_len, seq_len))  # tril
    for i in range(bs):
        pad_idx = torch.where(input_ids[i, :] == pad_token_id)[0]
        mask[i, pad_idx, :] = 0
        mask[i, :, pad_idx] = 0
    return mask


def get_src_trg_mask(src_ids, trg_ids,
                     src_pad_token_id=0,
                     trg_pad_token_id=0
                     ):
    bs, src_seq_len = src_ids.shape
    bs, trg_seq_len = trg_ids.shape

    mask = torch.ones(bs, trg_seq_len, src_seq_len)  # tril
    for i in range(bs):
        src_pad_idx = torch.where(src_ids[i, :] == src_pad_token_id)[0]
        trg_pad_idx = torch.where(trg_ids[i, :] == trg_pad_token_id)[0]
        mask[i, trg_pad_idx, :] = 0
        mask[i, :, src_pad_idx] = 0
    return mask


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--src_tokenizer_path",
                        type=str,
                        default='./output/tokenizer_zh',
                        # required=True
                        )
    parser.add_argument("--trg_tokenizer_path",
                        type=str,
                        default='./output/tokenizer_en',
                        # required=True
                        )
    parser.add_argument("--output_path",
                        type=str,
                        default='./output/transformer',
                        # required=True
                        )
    return parser
