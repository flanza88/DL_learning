import numpy as np

print("******3.3.1 多维数组******")
A = np.array([1, 2, 3, 4])
print(A)
print(A.ndim)
print(A.shape)#常用
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(B.ndim)
print(B.shape)#常用

print("******3.3.2 矩阵乘法******")
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))

A = np.array([[1, 2], [3, 4], [5,6]])
print(A.shape)
B = np.array([7, 8])
print(B.shape)
print(np.dot(A, B))#B看成2x1的形状

print("******3.3.3 神经网络的内积******")
X = np.array([1, 2])
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
Y = np.dot(X, W)#X看成1x2的形状
print(Y)

