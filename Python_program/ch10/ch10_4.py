import json

numbers = [2, 3, 5, 7, 11, 13]
filename = "numbers.json"
with open(filename, "w") as f:
    json.dump(numbers, f)

with open(filename) as f:
    numbers = json.load(f)
print(numbers)


def get_stored_username():
    filename = "username.json"
    try:
        with open(filename) as f:
            username = json.load(f)
    except FileNotFoundError:
        return None
    else:
        return username


def get_new_username():
    username = input("What is your name? ")
    filename = "username.json"
    with open(filename, "w") as f:
        json.dump(username, f)
    return username


def greet_user():
    username = get_stored_username()
    if username:
        print(f"Welcome back, {username}!")
    else:
        username = get_new_username()
        print(f"We'll remember you when you come back, {username}!")


greet_user()
