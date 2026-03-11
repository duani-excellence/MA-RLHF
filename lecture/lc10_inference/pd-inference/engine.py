from actor import RayActorGroup
from kvcache import DistributedKVCacheEngine
import ray
import torch
import random
import time
import asyncio


@ray.remote
class DissagreationPDEngine:
    def __init__(
        self,
        config,
        prefill_actor_group: RayActorGroup,
        decoding_actor_group: RayActorGroup,
        scheduler,  # update
        kvcache,
    ):
        self.prefill_actor_group = prefill_actor_group
        self.decoding_actor_group = decoding_actor_group
        self.is_running_prefill = True
        self.is_running_decoding = True

        self.kvcache = kvcache
        self.scheduler = scheduler

        self.config = config

    # def step(self, ):
    #     ray.get(self._step_decoding.remote())

    async def set_stop_prefill(self):
        self.is_running_prefill = False

    async def set_stop_decoding(self):
        self.is_running_decoding = False

    def _get_prefill_batch(self, info):
        bsz = len(info.ids)
        input_ids = torch.zeros(
            bsz, self.config.kv_cache_len, dtype=torch.long)
        for i in range(bsz):
            prompt = info.prompts[i]
            tmp_len_prompt = len(prompt)
            input_ids[i, :tmp_len_prompt] = torch.tensor(
                prompt, dtype=torch.long)
        return input_ids

    def _get_decoding_batch(self, info):
        bsz = len(info.ids)
        input_ids = torch.ones(bsz, 1, dtype=torch.long)
        for i in range(bsz):
            prompt = info.prompts[i]
            input_ids[i, 0] = torch.tensor(prompt[0], dtype=torch.long)

        return input_ids

    def _is_finish_prefill(self):
        result, running_prefill = ray.get(self.scheduler.get_num_pending_requests.remote())

        # if result == 0 and not self.is_running_prefill:
        if result == 0 and not running_prefill:
            print('[FINISH] Prefill Node')
            return True
        else:
            return False

    def _is_finish_decoding(self):
        result, running_decoding = ray.get(self.scheduler.get_num_running_requests.remote())
        # if len(result) == 0:
        #     time.sleep(0.1)

        # 非强制性结束 Decoding, 依赖 prefill 完成状态
        if result == 0 and not running_decoding:
            print('[FINISH] Decoding Node')
            return True
        else:
            return False

    def _step_decoding_process(self,):
        info = ray.get(self.scheduler.get_running_requests.remote())

        if info == None:
            time.sleep(random.random())
            return
        
        print('>>>>>> step decoding : ', info.ids)
        input_ids = self._get_decoding_batch(info)
        kv_cache = ray.get(self.kvcache.get_kv_cache.remote(info.ids))

        result = ray.get(self.decoding_actor_group.async_run_method(
            "forward", x=input_ids, kvcaches=kv_cache))
        # result = ray.get(refs)  # logits, kv

        logits = result[0][0]

        # 1. kvcache update, 2. request status update
        kv = result[0][1]
        ray.get(self.kvcache.update_from_decoding.remote(
            info.ids, kv, info.last_pos))
        
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # req_id, 1
        next_token = next_token.tolist()
        for i, req_id in enumerate(info.ids):
            ray.get(self.scheduler.update_request.remote(req_id, 
                                                  next_token[i])
                    )

    def _step_prefill_process(self,):
        info = ray.get(self.scheduler.get_waiting_requests.remote())
        if info == None:
            time.sleep(random.random())
            return

        input_ids = self._get_prefill_batch(info)

        # 这里获取的是多个 worker 的 results 值
        # prefill_actor_1: logits_1, kv_1
        # prefill_actor_2: logits_2, kv_2

        print('>>> step prefill : ', info.ids)
        results = ray.get(self.prefill_actor_group.async_run_method(
            "forward", x=input_ids))
        
        logits = results[0][0]  # 第一个 index 是 prefill actor id
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # req_id, 1
        
        # 1. kvcache update, 2. request status update
        ray.get(self.kvcache.update_from_prefill.remote(
            info.ids, results[0][1]))
        for i, req_id in enumerate(info.ids):
            ray.get(self.scheduler.update_request.remote(
                req_id, next_token[i].item()))

    def _step_prefill(self):
        while not self._is_finish_prefill():
            self._step_prefill_process()
            # await asyncio.sleep(0)
        print('-'*20, '[END] Preifll Task', '-'*20)
        time.sleep(2)
        return

    def _step_decoding(self):
        while not self._is_finish_decoding():
            self._step_decoding_process()
            # await asyncio.sleep(0)
        print('-'*20, '[END] Decoding Task', '-'*20)
        time.sleep(2)
        return
