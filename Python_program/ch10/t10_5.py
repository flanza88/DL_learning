filename = "reasons.txt"
while True:
    reason = input("为何喜欢编程")
    if reason == "q":
        break
    with open(filename, "a") as f:
        f.write(reason + "\n")
