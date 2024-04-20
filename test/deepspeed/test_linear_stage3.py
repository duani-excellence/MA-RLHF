# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.runtime.zero.linear import LinearModuleForZeroStage3
from deepspeed.utils.logging import logger
from deepspeed.accelerator import get_accelerator


# IN_DIM = 1024
# OUT_DIM = 16384
IN_DIM = 128
OUT_DIM = 1024

def see_memory_usage(message):
    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(
        "Memory Allocated %s GigaBytes ",
        get_accelerator().memory_allocated() / (IN_DIM * IN_DIM * IN_DIM),
    )
    logger.info(
        "Max Memory Allocated %s GigaBytes",
        get_accelerator().max_memory_allocated() / (IN_DIM * IN_DIM * IN_DIM),
    )
    logger.info(
        "Cache Allocated %s GigaBytes",
        get_accelerator().memory_cached() / (IN_DIM * IN_DIM * IN_DIM),
    )
    logger.info(
        "Max cache Allocated %s GigaBytes",
        get_accelerator().max_memory_cached() / (IN_DIM * IN_DIM * IN_DIM),
    )


tens = torch.rand(
    IN_DIM, OUT_DIM, dtype=torch.half, device=torch.device(get_accelerator().device_name())
)
tens_back = tens.detach().clone()

# linear_bk = torch.nn.functional.linear
# torch.nn.functional.linear = deepspeed.pt.deepspeed_linear.LinearFunctionForZeroStage3.apply
# model = LinearModuleForZeroStage3(16384, 16384)
model = LinearModuleForZeroStage3(OUT_DIM, OUT_DIM)

model.to(get_accelerator().device_name()).half()

see_memory_usage("Before forward")
y = model(tens)

see_memory_usage("After forward")

model.weight.data = torch.zeros(
    OUT_DIM,  OUT_DIM, dtype=torch.half, device=torch.device(get_accelerator().device_name())
)

see_memory_usage("After weight zero")

y.backward(tens_back)
