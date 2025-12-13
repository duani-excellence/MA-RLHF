from dataclasses import dataclass


EOS_TOKEN=0

@dataclass
class vLLMEngineConfig:
    max_batch_size = 4
    max_seq_len = 32
    max_prompt_len: int = 16
    max_new_tokens: int = 100

    # model 
    num_layers: int = 3
    dim: int = 16
    num_heads: int = 2
    head_dim: int = 8
    vocab_size: int = 20

    # PageKV Cache Setting
    page_size: int = 64
    num_pages: int = 1024

    # chunked-prefill
    max_batch_tokens: int = 2048
    max_decoding_batch: int = 16
    max_prefill_batch: int = 256



