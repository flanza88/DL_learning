pizzas = ['tomato', 'potato', 'apple', ]
friend_pizzas = pizzas[:]

pizzas.append('pepperoni')
friend_pizzas.append('mushroom')

print("My favorite pizzas are:")
for pizza in pizzas:
    print(pizza)
print("\nMy friend's favorite pizzas are:")
for pizza in friend_pizzas:
    print(pizza)