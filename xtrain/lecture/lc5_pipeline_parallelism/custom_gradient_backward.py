# python custom_gradient_backward.py
# 该例子为手动计算出梯度, 对中间变量反向传播, **而不是强制从`loss()`向上传播**
# 在 pipeline parallel里前向容易, 而反向其实要接受一个梯度矩阵在当前rank进行反向传播

import torch
import torch.nn as nn
import torch.autograd as autograd


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
label = torch.randn(2,4)
print(y.shape)
print(label.shape)


print('手动求gradient反向传播')
grad_y = 2 * (y-label) / y.numel()
y.backward(gradient = grad_y)

print(x.grad)
print(model.w.weight.grad)
x.grad = None
model.w.weight.grad = None


print('自动求gradient反向传播')
y = model(x)
loss_fn = nn.MSELoss()
loss = loss_fn(y, label)
loss.backward()
print(x.grad)
print(model.w.weight.grad)




