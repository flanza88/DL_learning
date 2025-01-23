motorcycles = ['honda', 'yamaha', 'suzuki', ]
print(motorcycles)

motorcycles[0] = 'ducati'
print(motorcycles)

motorcycles.append('shit')

motorcycles.insert(0, 'num1')

print(motorcycles)

del motorcycles[0]
popped_motorcycle = motorcycles.pop()
print(motorcycles)
print(popped_motorcycle)

first_owned = motorcycles.pop(0)
print(f"The first motorcycle I owned was a {first_owned.title()}")

motorcycles.remove('yamaha')
print(motorcycles)
