import torch

print("******2.5.1. 一个简单的例子******")
x = torch.arange(4.0)
print(x)

x.requires_grad_(True)#等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)

y = 2 * torch.dot(x,x)#y = 2 * x**2，梯度为4x
print(y)

y.backward()
print(x.grad)

print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()#y = x，梯度为1
y.backward()
print(x.grad)

print("******2.5.2. 非标量变量的反向传播******")
x.grad.zero_()
y = x**2
y.sum().backward()
print(x.grad)

print("******2.5.3. 分离计算******")
x.grad.zero_()
y = x**2
u = y.detach()#detach后y看成常数
z = u * x

z.sum().backward()
print(x.grad == u)

print("******2.5.4. Python控制流的梯度计算******")
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d / a)