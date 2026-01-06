import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from actor import Actor, BaseModelActor
from model import ToyModel


class PrefillActor(BaseModelActor):
    def __init__(self, decoding_actor):
        self.decoding_actor = decoding_actor
        self.reqs = []
        # def init

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

    # def _send_to_decoding_actor(self,):
    #     pass

    # def step(self,):
    #     if len(self.reqs) == 0:
    #         return

    #     self._get_batch()
    #     self.forward()
    #     self._send_to_decoding_actor()
