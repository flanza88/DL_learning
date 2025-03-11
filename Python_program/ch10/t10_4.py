filename = "guest_book.txt"
while True:
    name = input("请输入你的名字：")
    if name == "q":
        break
    with open(filename, "a") as f:
        f.write(name + "\n")
    print(f"Hi,{name}!")
