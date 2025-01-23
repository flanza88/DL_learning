print("输入一系列比萨配料,输入‘quit’退出")

while True:
    topping = input()
    if topping == 'quit':
        break
    else:
        print("好的,我们会添加" + topping + "配料")

