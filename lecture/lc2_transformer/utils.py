import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)

def get_src_mask(input_ids, pad_token_id = 0):
    bs, seq_len = input_ids.shape
    mask = torch.ones(bs, seq_len, seq_len)
    for i in range(bs):
        pad_idx =  torch.where(input_ids[i, :]  == pad_token_id)[0]
        mask[i, pad_idx, :] = 0
        mask[i, :, pad_idx] = 0
    return mask

def get_trg_mask(input_ids, pad_token_id = 0):
    bs, seq_len = input_ids.shape
    mask = torch.tril(torch.ones(bs, seq_len, seq_len)) # tril
    for i in range(bs):
        pad_idx =  torch.where(input_ids[i, :]  == pad_token_id)[0]
        mask[i, pad_idx, :] = 0
        mask[i, :, pad_idx] = 0
    return mask

def get_src_trg_mask(src_ids, trg_ids, pad_token_id = 0):
    bs, src_seq_len = src_ids.shape
    bs, trg_seq_len = trg_ids.shape
    
    mask = torch.ones(bs, trg_seq_len, src_seq_len) # tril
    for i in range(bs):
        src_pad_idx =  torch.where(src_ids[i, :]  == pad_token_id)[0]
        trg_pad_idx =  torch.where(trg_ids[i, :]  == pad_token_id)[0]
        mask[i, trg_pad_idx, :] = 0
        mask[i, :, src_pad_idx] = 0
    return mask