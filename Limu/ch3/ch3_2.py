# %matplotlib inline
import random
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

print("******3.2.1. 生成数据集******")
def synthetic_data(w, b, num_examples):
    """生成y=Wx+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
#features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）
features, labels = synthetic_data(true_w, true_b, 1000)

print("feature[0]:", features[0], '\nlabel[0]:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
plt.show()

print("******3.2.2. 读取数据集******")
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

print("******3.2.3. 初始化模型参数******")
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print("******3.2.4. 定义模型******")
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

print("******3.2.5. 定义损失函数******")
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

print("******3.2.6. 定义优化算法******")
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

print("******3.2.7. 训练******")
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):#一个批次的梯度下降
        l = loss(net(X, w, b), y)
        l.sum().backward()#反向传播
        sgd([w, b], lr, batch_size)#更新参数，梯度清零
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')