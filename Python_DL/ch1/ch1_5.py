import numpy as np
print("******1.5 Numpy******")

x = np.array([1., 2., 3.])
print(x)
print(type(x))

x = np.array([1., 2., 3.])
y = np.array([2., 4., 6.])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x / 2.0)

A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)
B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)
print(A * 10)

print("******1.5.5 广播******")
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

X = np.array([[51,55], [14,19], [0,4]])
print(X)
print(X[0])
print(X[0,1])#print(X[0][1])

for row in X:
    print(row)

X = X.flatten()#将X转换为一维数组
print(X)
print(X[np.array([0,2,4])])
print(X > 15)
print(X[X > 15])#取True的值