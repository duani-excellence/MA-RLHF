from actor import RayActorGroup
from kvcache import DistributedKVCacheEngine
import ray
import torch
import random
import time


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

    def set_stop_prefill(self):
        self.is_running_prefill = False

    def set_stop_decoding(self):
        self.is_running_decoding = False

    def _get_prefill_batch(self, info):
        bsz = len(info)
        input_ids = torch.zeros(
            bsz, self.config.kv_cache_len, dtype=torch.long)
        for i in range(bsz):
            prompt = info.prompts[i]
            tmp_len_prompt = len(prompt)
            input_ids[i, :tmp_len_prompt] = prompt

        return input_ids

    def _get_decoding_batch(self, info):
        bsz = len(info)
        input_ids = torch.zeros(bsz, 1, dtype=torch.long)
        for i in range(bsz):
            prompt = info.prompts[i]
            # tmp_len_prompt = len(prompt)
            input_ids[i, 0] = prompt[0]

        return input_ids

    def _is_finish_prefill(self):
        result = ray.get(self.scheduler.get_num_pending_requests.remote())

        if result == 0 and not self.is_running_prefill:
            return True
        else:
            return False

    def _is_finish_decoding(self):
        result = ray.get(self.scheduler.get_num_running_requests.remote())

        # 非强制性结束 Decoding, 依赖 prefill 完成状态
        if result == 0 and self._is_finish_prefill() and not self.is_running_decoding:
            return True
        else:
            return False

    def _step_prefill(self):
        while not self._is_finish_prefill():
            info = ray.get(self.scheduler.get_waiting_requests.remote())
            if info == None:
                time.sleep(random.random())
            input_ids = self._get_prefill_batch(info.ids)
            refs = ray.get(self.prefill_actor_group.async_run_method(
                "forward", x=input_ids))
            result = ray.get(refs)  # logits, kv

            logits = result[0]
            next_token = torch.argmax(logits[:, -1, :])
            for i, req_id in enumerate(info.ids):
                self.scheduler.update_request(req_id, next_token[i])

            # 将 KV 传输到 decoding 节点的 cache 里
            ray.get(self.kvcache.update_from_prefill.remote(
                info.ids, result[1]))
        return

    def _step_decoding(self):
        while not self._is_finish_decoding():
            info = self.scheduler.get_running_requests()
            if info == None:
                time.sleep(random.random())

            input_ids = self._get_decoding_batch(info.ids)
            kv_cache = ray.get(self.kvcache.get_kv_cache.remote(info.ids))

            refs = ray.get(self.decoding_actor_group.async_run_method(
                "forward", x=input_ids, kvcaches=kv_cache))
            result = ray.get(refs)  # logits, kv

            logits = result[0]

            next_token = torch.argmax(logits[:, -1, :])
            for i, req_id in enumerate(info.ids):
                self.scheduler.update_request(req_id, next_token[i])

            kv = result[1]
            ray.get(self.kvcache.update_from_decoding.remote(
                info.ids, kv, info.last_pos))

        return
