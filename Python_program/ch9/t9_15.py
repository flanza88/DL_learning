import random

list0 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e"]
my_ticket = []
while True:
    a = list0[random.randint(0, 14)]
    b = list0[random.randint(0, 14)]
    c = list0[random.randint(0, 14)]
    d = list0[random.randint(0, 14)]
    my_ticket.append(a + b + c + d)
    if my_ticket[-1] == "0a1b":
        print(f"循环{len(my_ticket)}次才中了大奖")
        break
