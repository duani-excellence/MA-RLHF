import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

def token_pre_process(token_ids_list, 
                      sos_token_id = None, 
                      eos_token_id = None):
    token_ids_pre_process = []
    for token_ids in token_ids_list:
        if sos_token_id is not None:
            token_ids = [sos_token_id] + token_ids
        if eos_token_id is not None:
            token_ids = token_ids + [eos_token_id] 
        token_ids_pre_process.append(token_ids)
    return token_ids_pre_process

def padding_max_length(input_ids, 
                       max_len = 32, 
                       pad_token_id = None, 
                       padding_side = 'RIGHT',
                      truction_side = 'RIGHT'):
    if pad_token_id is None:
        return
    tokens_lens = [ len(ids) for ids in input_ids]
    tokens_lens = torch.tensor(tokens_lens, dtype = torch.long)
    tokens_max_len = torch.max(tokens_lens)

    # trunction
    if tokens_max_len > max_len:
        tokens_max_len = max_len
    if truction_side == 'RIGHT':
        input_ids = [ ids[ : min(len(ids), tokens_max_len)] for ids in input_ids]
    else:
        input_ids = [ ids[ -min(len(ids), tokens_max_len) : ] for ids in input_ids]

    # padding
    paddding_input_ids = torch.ones(len(input_ids), tokens_max_len, dtype = torch.long) * pad_token_id
    if padding_side == 'RIGHT':
        for i in range(len(input_ids)):
            paddding_input_ids[i, : len(input_ids[i])] = torch.tensor(input_ids[i], dtype = torch.long)
    else: # left padding
        for i in range(len(input_ids)):
            paddding_input_ids[i, -len(input_ids[i]):] = torch.tensor(input_ids[i], dtype = torch.long)

    return paddding_input_ids