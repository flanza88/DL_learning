usr = input("输入名字")
filename = "guest.txt"
with open(filename, "w") as f:
    f.write(usr + "\n")
