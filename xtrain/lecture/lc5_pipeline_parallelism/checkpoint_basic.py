import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

'''
基础的checkpoint用法
`use_reentrant=True` 则计算过程不记录在计算图里, `False` 则记录在计算图里
'''
def my_function(x, y):
    return x * y

x = torch.randn(1, 10, requires_grad=True)
y = torch.randn(1, 10, requires_grad=True)

# 使用 checkpoint
output = checkpoint(my_function, x, y, use_reentrant=False)
print(output)

loss = (output ** 2).mean()
loss.backward()
print(x.grad)

'''
基于`gradient checkpoint` 训练
ref: `https://gist.github.com/vanbasten23/94b33839fa37469f30a03487d1fac746`
'''
def checkpointed_forward(module, *inputs):
    return checkpoint(module, *inputs, use_reentrant=False)

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        self.layer3 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = checkpointed_forward(self.layer2, x)
        x = self.layer3(x)
        return x
       
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

inputs = torch.randn(8,10)
labels = torch.randn(8,10)

# Example training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    print(loss)
    optimizer.step()
