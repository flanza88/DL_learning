file_name = "learning_python.txt"
with open(file_name) as f:
    contents = f.read()
    contents = contents.replace("Python", "C")
print(contents)

with open(file_name) as f:
    for line in f:
        line = line.replace("Python", "C")
        print(line.strip())

list_str = []
with open(file_name) as f:
    for line in f:
        line = line.replace("Python", "C")
        list_str.append(line.strip())
print(list_str)
