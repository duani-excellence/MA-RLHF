import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import json
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


def save_dict_to_json(filepath, data: dict = None):
    """将字典保存为 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"字典已保存为 JSON 文件: {filepath}")
