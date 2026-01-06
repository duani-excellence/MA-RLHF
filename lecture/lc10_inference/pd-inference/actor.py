# Reference: [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed


import os
import socket
from typing import Dict, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm

from model import ToyModel
from utils import ray_noset_visible_devices


class Actor(nn.Module):
    def __init__(self, config, model_type):
        self.model = model_type(config)

    def forward(self, x, kvcaches=None, current_length=None):
        # 改成传参列表 **param
        logits, kv = self.model(x, kvcaches, current_length)
        return logits, kv


class BaseDistributedActor:
    def __init__(self, world_size, rank, master_addr, master_port):

        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(
            ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    def is_rank_0(self,):
        return self._rank == 0


class BaseModelActor(BaseDistributedActor):

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self, method_name: str, all_data, start_idx, end_idx):
        """Process input data by calling specified function for each item in the lists.

        Args:
            method_name (str): Name of the function to execute
            kwargs: Reference to the chunk of data to process

        Returns:
            List[Any]: List of results from function execution
        """

        # Get the first parameter to determine list length
        kwargs = {key: value[start_idx:end_idx]
                  for key, value in all_data.items()}
        first_param = next(iter(kwargs.values()))
        list_length = len(first_param)

        # Get the function to execute
        func = getattr(self, method_name)
        if not callable(func):
            raise ValueError(f"Function {method_name} is not callable")

        results = []
        for i in tqdm(range(list_length), desc=f"{method_name}", disable=self.is_rank_0()):
            # Create kwargs for single item
            sample_kwargs = {param_name: param_value[i]
                             for param_name, param_value in kwargs.items()}

            result = func(**sample_kwargs)
            results.append(result)

        return results


class RayActorGroup:
    # 存在 case:
    #   1. 一个 actor, 分布在多个 GPU
    #   2. 多个 actor, 分布在多个 GPU
    #   3. 多个 atorc, 分布在一个 GPU

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BaseModelActor],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        duplicate_actors: int = 1,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        self.duplicate_actors = duplicate_actors
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            # bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            bundles = [{"CPU": 1}
                       for _ in range(self._num_nodes * self._num_gpus_per_node)]
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())

        master_actor = self.ray_actor_type.options(
            num_cpus=num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker_actor
        if world_size > 1:
            master_addr, master_port = ray.get(
                master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                worker_actor = self.ray_actor_type.options(
                    num_cpus=num_gpus_per_actor,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=rank,
                    ),
                ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs
