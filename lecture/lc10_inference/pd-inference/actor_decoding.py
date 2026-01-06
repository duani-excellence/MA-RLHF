import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from actor import Actor, RayActorGroup, BaseModelActor
from model import ToyModel


class DecodingActor(BaseModelActor):
    # 注意操作 master actor 和 worker actor
    def __init__(self, decoding_actor):
        self.decoding_actor = decoding_actor
        self.reqs = []

    def server_start(self,):
        # init_dist()
        pass

    def from_pretrained(self, config, model_type):
        self.actor = Actor(
            config,
            model_type
        )

    def forward(self, **param):
        logits, kv = self.actor(param)
        return logits, kv

    # def _get_batch(self,):
    #     pass

    # def _from_prefill_actor(self,):
    #     # 更新 reqs
    #     # 更新 kvcache
    #     # 接收的数据存储至 buffer
    #     pass

    # def _set_kvcache(self,):
    #     # 创建一个 kvcache 对象, 用于异步接收 cache
    #     pass

    # def update(self, ):
    #     # 基于 forward 结果个更新 reqs
    #     # 基于请求状态更新 kv cache
    #     # 将 cache buffer 与最新的 reqs 进行融合
    #     pass

    # def step(self,):
    #     if len(self.reqs) == 0:
    #         return

    #     self._get_batch()
    #     self.forward()
    #     self.update()
