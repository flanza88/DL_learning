import torch

print("******2.3.1. 标量******")
x = torch.tensor(3.)
y = torch.tensor(2.)

print(x + y, x * y, x / y, x ** y)

print("******2.3.2. 向量******")
x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

print("******2.3.3. 矩阵******")
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1,2,3], [2,0,4], [3,4,5]])
print(B)
print(B == B.T)

print("******2.3.4. 张量******")
X = torch.arange(24).reshape(2, 3, 4)
print(X)

print("******2.3.5. 张量算法的基本性质******")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A, A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

print("******2.3.6. 降维******")
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
print(A.shape, A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))

print(A.mean(), A.sum() / A.numel())

print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

sum_A = A.sum(axis=1, keepdim=True)
print(sum_A, sum_A.shape)
print(A / sum_A)

print(A.cumsum(axis=0))

print("******2.3.7. 点积（Dot Product）******")
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))

print("******2.3.8. 矩阵-向量积******")
print(A.shape, x.shape, torch.mv(A, x))

print("******2.3.9. 矩阵-矩阵乘法******")
B = torch.ones(4,3)
print(torch.mm(A, B))

print("******2.3.10. 范数******")
u = torch.tensor([3, -4.])
print(torch.norm(u))#L2
print(torch.abs(u).sum())#L1
