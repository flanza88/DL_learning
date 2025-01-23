peoples = ['mike', 'john', 'lucy']
for people in peoples:
    print(f"{people.title()} come to eat with me")

print("I find a bigger tabe")

peoples.insert(0,'num1')
peoples.insert(2,'middle')
peoples.append('last')
for people in peoples:
    print(f"{people.title()} come to eat with me")

print("I can only invite 2 peoples")

while len(peoples) > 2:
    people = peoples.pop()
    print(f"sorry {people.title()} I can not invite you")

for people in peoples:
    print(f"{people.title()} come to eat with me")

del peoples[0]
del peoples[0]
print(peoples)


