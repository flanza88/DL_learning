file_name = "learning_python.txt"
with open(file_name) as f:
    contents = f.read()
print(contents)

with open(file_name) as f:
    for line in f:
        print(line.strip())

list_str = []
with open(file_name) as f:
    for line in f:
        list_str.append(line.strip())
print(list_str)
