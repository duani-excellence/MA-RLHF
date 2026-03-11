# python zero_bubble_seperate_dx_dw.py
'''
ref: https://github.com/deepseek-ai/DualPipe
'''

from typing import List, Optional, Callable, Tuple
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


class WeightGradStore:
    '''
    全局变量, 管理对w的backward, 在执行pop()时完成对队列所有元素求导
    '''
    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()

class LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = F.linear(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)

        def grad_weight_fn():
            '''
            以下打印表明, 这些中间变量会存储下来, 实际上也需要占用显存
            '''
            # print('[grad_weight_fn], grad_output', grad_output[0, :4])
            # print('[grad_weight_fn], input', input[0, :4])
            weight.grad += grad_output.flatten(0, -2).T @ input.flatten(0, -2)

        if WeightGradStore.enabled:
            WeightGradStore.put(grad_weight_fn)
        else:
            grad_weight_fn()
        grad_input = grad_output @ weight
        return grad_input, None

    @classmethod
    def overlapped_forward_backward(
        cls,
        module0: "PipelineStage",
        inputs0: List[torch.Tensor],
        criterion0: Optional[Callable],
        labels0: Optional[List[torch.Tensor]],
        module1: "PipelineStage",
        loss1: Optional[torch.Tensor],
        outputs1: Optional[List[torch.Tensor]],
        output_grads1: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        You should implement custom forward-backward overlap strategy.
        The code below is just an example.
        """
        outputs0 = module0(*inputs0)
        outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
        if criterion0 is not None:
            loss0 = criterion0(*outputs0, *labels0)
        else:
            loss0 = None

        if loss1 is not None:
            loss1.backward()
            loss1.detach_()
        else:
            run_backward(outputs1, output_grads1)

        return outputs0, loss0



class MyLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LinearFunc.apply(input, self.weight)


class PipelineStage(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = MyLinear(hidden_size, hidden_size * 4, bias=False)
        self.linear2 = MyLinear(hidden_size * 4, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x

# dim = 4
# model = PipelineStage(hidden_size=4)
num_layer = 2
bs = 8
dim = 4
input = torch.randn(bs, dim, requires_grad=True)
label = torch.randn(bs, dim)
loss_fn = nn.MSELoss()

WeightGradStore.enabled = True
model = nn.Sequential()
for i in range(num_layer):
    model.append(PipelineStage(hidden_size=4))
print(model)
print('forward')
output = model(input)
loss = loss_fn(output, label)

print('linear grad:', model[0].linear1.weight.grad)
print('backward')
loss.backward()

print('linear grad:', model[0].linear1.weight.grad)
print('WeightGradStore.cache:', WeightGradStore.cache)
print('WeightGradStore.funcs_queue:', WeightGradStore.funcs_queue)

print('flush')
WeightGradStore.flush()
print('WeightGradStore.cache:', WeightGradStore.cache)
print('WeightGradStore.funcs_queue:', WeightGradStore.funcs_queue)

print('pop')
WeightGradStore.pop()
print('linear grad:', model[0].linear1.weight.grad)

