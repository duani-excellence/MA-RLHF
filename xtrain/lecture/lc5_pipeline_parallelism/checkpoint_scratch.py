# python checkpoint_scratch.py

# prompt: 详细分析pytorch里关于torch.utils.checkpoint模块的实现，如何干预计算图从而在autograd过程可以进行重计算，如何提取计算过程中的中间变量。
# 基于 deepseek-chat 回复进行修改

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from pipeline_parallel_basic import MLP

class CustomCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, *args):
        ctx.func = func
        
        print('前向时的网络信息:', func)
        print('输入参数为:', args)

        ctx.save_for_backward(*args)
        with torch.no_grad():
            '''
            不会有中间的 activation 保存在计算图里
            '''
            outputs = func(*args)
        return outputs
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        print('backward parameteres inputs:',input)

        # ReCompute
        # gradient checkpoint 
        with torch.enable_grad():
            outputs = ctx.func(*inputs) # 这个过程执行了w1和w2 linear前向计算
        
        # 计算梯度并返回
        torch.autograd.backward(outputs, grad_outputs)

        return (None, ) + tuple(inp.grad for inp in inputs)
    

dim = 3
N = 2
model = MLP(dim, rank = 0, world_size=1)
input = torch.randn(N, dim, requires_grad=True)
label = torch.randn(N, dim)

intermediates = []
def hook(module, input, output):
    '''
    用于检测模型是否进入了重计算
    对这层前向过一遍则会记录前4个元素, 理论上涉及到重计算则会前向两遍
    '''
    print('前向 hook output shape:', output.shape)
    intermediates.append(output[0,:4])
model.w1.register_forward_hook(hook) # 预期intermediates会存储两个一样的 2x12 大小的矩阵

print('-'*20, 'forward')
output = CustomCheckpointFunction.apply(model.forward, input)
print('前向中间变量为: ', intermediates)
print(output.shape)

print('-'*20, 'backward')
grad_output = 2 * (output-label) / label.numel()
output.backward(gradient=grad_output)
print('d_input:', input.grad.shape) # 2 x 3
print('w1:', model.w1.weight.grad.shape) # 2 x 12
print('w2:', model.w2.weight.grad.shape) # 12 x 3
print('后向中间变量为: ', intermediates) # 重计算时的数据是有 `grad_fn` 的