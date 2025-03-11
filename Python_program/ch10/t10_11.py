import json

filename = "favorite_number.json"
try:
    with open(filename) as f:
        favorite_number = json.load(f)
        print(f"I know your favorite number! It's {favorite_number}")
except FileNotFoundError:
    favorite_number = int(input("What is your favorite number? "))
    with open(filename, "w") as f:
        json.dump(favorite_number, f)
else:
    pass
