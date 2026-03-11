# python tensor_parallel.py

import torch
import torch.nn as nn
import torch.autograd as autograd

class MyFuntion(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # output = ctx.w(input)
        output = input * 2
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input * 3 
        return grad_input 
    
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 8)

    def forward(self, x):
        return MyFuntion.apply(x)   
    
model = MyModel()
x = torch.randn(1, 4, requires_grad=True)
y = model(x)
y.sum().backward()
print(x.grad)


# 在数据并行时, 我们手动实现了梯度的reduce
# 而在张量并行时, 由于对参数进行了切分, 就不一定能像 数据并行 时一样能后置来 reduce梯度
# 所以我们如果能手动定义反向方法, 则能在反向过程里实现分布式操作

class MyGradFuntion(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        output = input @ w 
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input =  grad_output @ w.t() 
        grad_w = input.t() @ grad_output 
        return  grad_input, grad_w 
    
class MyGradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(3, 4, bias = False)

    def forward(self, x):
        return MyGradFuntion.apply(x, self.w.weight.t())   
    
model = MyGradModel()
x = torch.randn(2, 3, requires_grad=True)
y = model(x)
y.sum().backward()
print(x.grad)
print(model.w.weight)
print(model.w.weight.grad)
