from config import vLLMEngineConfig
from model import PageToyModel
from engine import vLLMEngine

import torch
import random
torch.manual_seed(42)
random.seed(42)


def listen_request(config, p=0.01):
    prompt = []
    prompt_len = 0
    num = random.randint(1, 100)
    if num/100.0 < p:
        prompt_len = random.randint(
            config.max_prompt_len//4, config.max_prompt_len)
        prompt = torch.randint(
            low=1, high=config.vocab_size, size=(1, prompt_len))
        prompt = prompt[0].tolist()
    return prompt, prompt_len


N = 64
count = 0

config = vLLMEngineConfig()
model = PageToyModel(config)
engine = vLLMEngine(model, config)

while 1:
    # 监听进程, 可能获取多条请求
    # if count != N:
    if count <= N:
        for i in range(5):  # 最多获取 5 条数据:
            prompt, prompt_len = listen_request(config, p=0.4)
            if prompt_len != 0:
                count += 1
                if count % (N//10) == 0:
                    per = count / (N//10)
                    print('Running...:', '*'*int(per), '-'*(10-int(per)))
                generate_len = random.randint(
                    config.max_new_tokens//4, config.max_new_tokens)
                engine.add_request(prompt, generate_len)

    # 处理进程
    with torch.no_grad():
        engine.step(config=config)

    if not engine.has_pending_work() and count > N:
        print('process done', count)
        break
