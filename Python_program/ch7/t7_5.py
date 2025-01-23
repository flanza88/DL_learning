
while True:
    age = input("How old?")
    if age == 'quit':
        break
    age = int(age)
    if age < 3:
        print("Free")
    elif 3 <= age < 12:
        print("$10")
    else:
        print("15")

