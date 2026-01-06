from scheduler import Scheduler
from kvcache import DistributedKVCacheEngine
from actor_decoding import DecodingActor
from actor_prefill import PrefillActor
from engine import DissagreationPDEngine
from config import PDInferenceEngineConfig
from actor import RayActorGroup
import ray
import torch
import random
import time

ray.init()


def get_requests(config):
    prompt = []
    prompt_len = 0
    prompt_len = random.randint(
        config.max_prompt_len//4, config.max_prompt_len)
    prompt = torch.randint(
        low=1, high=config.vocab_size, size=(1, prompt_len))
    prompt = prompt[0].tolist()
    generate_len = random.randint(
        config.max_new_tokens//4, config.max_new_tokens)
    return prompt, generate_len


@ray.remote
def sender(config, scheduler, engine, max_prompts):

    for i in range(max_prompts):
        prompt, gen_len = get_requests(config)
        ray.get(scheduler.add_request.remote(prompt, gen_len))
        time.sleep(random.randint(1, 5))

    ray.get(engine.set_stop_prefill.remote())
    ray.get(engine.set_stop_decoding.remote())
    return


def train(config):

    scheduler = Scheduler.remote(config)

    kvcache = DistributedKVCacheEngine.remote(config)

    decoding_actor = RayActorGroup(
        1,
        1,
        DecodingActor,
    )
    prefill_actor = RayActorGroup(
        1,
        1,
        DecodingActor,
    )
    DissagreationPDEngine(
        config=config,
        prefill_actor_group=prefill_actor,
        decoding_actor_group=decoding_actor,
        scheduler=scheduler,  # update
        kvcache=kvcache,
    )

    ray.get(sender.remote(config, scheduler, engine, config.max_prompts))

    # start endless step compute task
    ray.get(DissagreationPDEngine._step_prefill.remote())
    ray.get(DissagreationPDEngine._step_decoding.remote())

    return


if __name__ == '__main__':
    config = PDInferenceEngineConfig()
    train(config)
