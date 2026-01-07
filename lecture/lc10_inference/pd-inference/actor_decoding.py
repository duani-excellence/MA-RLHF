import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from actor import Actor, RayActorGroup, BaseModelActor
from model import ToyModel
import ray

@ray.remote(num_cpus=1)
class DecodingActor(BaseModelActor):
    # 注意操作 master actor 和 worker actor
    # def __init__(self, decoding_actor):
    #     self.decoding_actor = decoding_actor
    #     self.reqs = []
    

    def server_start(self,):
        # init_dist()
        pass

    def init_model_from_pretrained(self, config, model_type):
        self.actor = Actor(
            config,
            model_type
        )

    def forward(self, x, kvcaches=None, current_length=None):
        logits, kv = self.actor(x=x, 
                                kvcaches=kvcaches, 
                                current_length=current_length)
        return logits, kv
