palces = dict()

while True:
    user_places = list()
    user= input("请输入用户名：")
    while True:
        place = input("请输入你喜欢的地方：,输入quit结束")
        if place == 'quit':
            break
        else:
            user_places.append(place)
    palces[user] = user_places

    repeat = input("是否继续调查：(yes/no)")
    if repeat == 'no':
        break

print("--RESULT--")
for user, places in palces.items():
    print(f'{user}喜欢的地方有：{places}')