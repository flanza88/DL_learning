import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

print("******3.7.1. 初始化模型参数******")
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)#apply 方法递归地遍历模型中的每一个子模块，并对每个子模块调用 init_weights 函数

print("******3.7.2. 重新审视Softmax的实现******")
loss = nn.CrossEntropyLoss(reduction='none')

print("******3.7.3. 优化算法******")
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

print("******3.7.4. 训练******")
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()