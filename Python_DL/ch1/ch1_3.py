print("******1.3.1 算术计算******")
print(1 - 2)
print(4 * 5)
print(7 / 5)
print(3 ** 2)

print("******1.3.2 数据类型******")
print(type(10))
print(type(2.718))
print(type("hello"))

print("******1.3.3 变量******")
x = 10
print(x)
x = 100
print(x)
y = 3.14
print(x * y)
print(type(x * y))

print("******1.3.4 列表******")
a = [1, 2, 3, 4, 5]
print(a)
print(len(a))
print(a[0])
print(a[4])
a[4] = 99
print(a)
print(a[0:2])
print(a[1:])
print(a[:3])
print(a[:-1])
print(a[:-2])

print("******1.3.5 字典******")
me = {"height" : 190}
print(me["height"])
me["weight"] = 80
print(me)

print("******1.3.6 布尔型******")
hungry = True
sleepy = False
print(type(hungry))
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

print("******1.3.7 if语句******")
hungry = True
if hungry:
    print("I'm hungry")

hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")

print("******1.3.8 for语句******")
for i in [1, 2, 3]:
    print(i)

print("******1.3.9 函数******")
def hello():
    print("hello world")
hello()

def hello(object):
    print("Hello " + object + "!")
hello("cat")
