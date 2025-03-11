import numpy as np
import matplotlib.pyplot as plt
"""
激活函数h()：
a = b + w1*x1 + w2*x2
y = h(a)
"""

print("******3.2.2 阶跃函数的实现******")
def step_function(x):
    y = x > 0
    return y.astype(np.int64)#bool转int64

x = np.array([-1.0, 1.0, 2.0])
print(x)
y = x > 0
print(y)
y = y.astype(np.int64)
print(y)

def step_function(x):
    return np.array(x > 0, dtype=np.int64)#bool转int64

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

print("******3.2.4 sigmoid函数的实现******")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

print("******3.2.7 ReLU函数******")
def relu(x):
    return np.maximum(0, x)
