filename = "dogs.txt"
try:
    with open(filename) as f:
        contents = f.read()
        print(contents.strip())
except FileNotFoundError:
    pass
