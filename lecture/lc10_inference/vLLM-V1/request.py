import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from .config import EOS_TOKEN

class Request:
    def __init__(self,
                 request_id: int,
                 prompt: List[int],
                 max_len: int = 2048):
        self.request_id = request_id
        self.prompt = prompt
        self.generated_tokens = []
        self.status = "REQUEST_WAITING"  # WAITING, RUNNING, COMPLETED
        self.current_length = len(prompt)
        self.max_length = max_len

    def add_token(self, token: int):
        """添加生成的token到请求中"""
        self.generated_tokens.append(token)
        self.current_length += 1
        if self.is_finished():
            self.status = "REQUEST_COMPLETED"
            print(
                f'finished: ID.{self.request_id}, new_len:{len(self.generated_tokens)}')

    def is_finished(self) -> bool:
        """检查请求是否完成(达到最大长度或生成了EOS)"""
        result =  bool(self.current_length >= self.max_length or (
            self.generated_tokens and self.generated_tokens[-1] == EOS_TOKEN))
        return result

    def get_full_sequence(self) -> List[int]:
        """获取完整的序列(prompt + 生成的tokens)"""
        return self.prompt + self.generated_tokens