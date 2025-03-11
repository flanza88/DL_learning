import random

list0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e"]
a = list0[random.randint(0, 14)]
b = list0[random.randint(0, 14)]
c = list0[random.randint(0, 14)]
d = list0[random.randint(0, 14)]
print(a)
print(b)
print(c)
print(d)
if a == 0 and b == 1 and c == "a" and d == "b":
    print("bingo")
