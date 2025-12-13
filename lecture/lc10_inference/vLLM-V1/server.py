from .config import vLLMEngineConfig
from .model import PageToyModel
from .engine import vLLMEngine

N = 32
count = 0


config = vLLMEngineConfig()
model = PageToyModel(config)
engine = vLLMEngine(model, config)

# main
while 1:
    # 处理进程
    engine.step()

    if not engine.has_pending_work() and count == N:
        print('process done')
        break