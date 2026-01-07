# by xiaodongguaAIGC

from scheduler import Scheduler
from kvcache import DistributedKVCacheEngine
from actor_decoding import DecodingActor
from actor_prefill import PrefillActor
from engine import DissagreationPDEngine
from config import PDInferenceEngineConfig
from actor import RayActorGroup
from model import ToyModel

import ray
from ray.util.placement_group import placement_group
import torch
import random
import time
import asyncio

ray.init()


def get_requests(config):
    prompt = []
    prompt_len = 0
    prompt_len = random.randint(
        config.max_prompt_len//4, config.max_prompt_len)
    prompt = torch.randint(
        low=1, high=config.vocab_size, size=(1, prompt_len))[0].tolist()
    generate_len = random.randint(
        config.max_new_tokens//4, config.max_new_tokens)
    return prompt, generate_len


@ray.remote
def sender(config, scheduler, engine_prefill, engine_decoding, max_prompts):

    for i in range(max_prompts):
        prompt, gen_len = get_requests(config)
        ray.get(scheduler.add_request.remote(prompt, gen_len))
        print(f'<<< Send Request, no.{i}')
        time.sleep(random.random())

    print(f'<<< Send Stop Signal')
    ray.get(scheduler.set_stop_prefill.remote())
    ray.get(scheduler.set_stop_decoding.remote())
    print(f'<<< Send Stop Signal end')
    
    return


def server(config):

    scheduler = Scheduler.remote(config)
    kvcache = DistributedKVCacheEngine.remote(config)

    # 创建 decoding 节点
    pg = None
    bundles = [{"CPU": 1} for _ in range(config.actor_group_gpu)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    decoding_actor = RayActorGroup(
        num_nodes=1,
        num_gpus_per_node=config.actor_group_gpu,
        ray_actor_type=DecodingActor,
        pg=pg,
        num_gpus_per_actor=1,
    )
    
    # 创建 prefill 节点    
    pg = None
    bundles = [{"CPU": 1} for _ in range(config.actor_group_gpu)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    prefill_actor = RayActorGroup(
        num_nodes=1,
        num_gpus_per_node=config.actor_group_gpu,
        ray_actor_type=PrefillActor,
        pg=pg,
        num_gpus_per_actor=1,
    )
    
    prefill_engine = DissagreationPDEngine.remote(
        config=config,
        prefill_actor_group=prefill_actor,
        decoding_actor_group=None,
        scheduler=scheduler,  # update
        kvcache=kvcache,
    )
    decoding_engine = DissagreationPDEngine.remote(
        config=config,
        prefill_actor_group=None,
        decoding_actor_group=decoding_actor,
        scheduler=scheduler,  # update
        kvcache=kvcache,
    )

    ray.get(prefill_actor.async_init_model_from_pretrained(config, ToyModel))
    ray.get(decoding_actor.async_init_model_from_pretrained(config, ToyModel))
    
    future_sender = sender.remote(
        config, 
        scheduler, 
        prefill_engine, 
        decoding_engine,
        config.max_prompts)

    # start endless step compute task
    # future_1 = ray.get(engine._step_prefill.remote())
    future_1 = prefill_engine._step_prefill.remote()
    furure_2 = decoding_engine._step_decoding.remote()

    ray.get(future_1)
    ray.get(furure_2)
    ray.get(future_sender)
    
    time.sleep(5)

    return


if __name__ == '__main__':
    config = PDInferenceEngineConfig()
    server(config)
