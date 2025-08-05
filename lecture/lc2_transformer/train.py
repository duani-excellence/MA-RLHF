import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tokenizer import TokenizerBase, TokenizerBaseConfig
from model import Transformer
from utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

torch.manual_seed(42)

if __name__ == "__main__":

    # load tokenizer
    tokenizer_en = TokenizerBase()
    tokenizer_zh = TokenizerBase()

    # create model

    # load dataset

    # process dataset

    # train

    # evaluation

    # save