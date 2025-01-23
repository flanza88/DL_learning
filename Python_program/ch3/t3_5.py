peoples = ['mike', 'john', 'lucy']
for people in peoples:
    print(f"{people.title()} come to eat with me")

print("mike can not come")

peoples.remove('mike')
for people in peoples:
    print(f"{people.title()} come to eat with me")



