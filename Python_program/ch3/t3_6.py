peoples = ['mike', 'john', 'lucy']
for people in peoples:
    print(f"{people.title()} come to eat with me")

print("I find a bigger tabe")

peoples.insert(0,'num1')
peoples.insert(2,'middle')
peoples.append('last')
for people in peoples:
    print(f"{people.title()} come to eat with me")



