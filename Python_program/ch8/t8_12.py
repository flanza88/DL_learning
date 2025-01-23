

def add(*toppings):
    print("Your sandwich includes:")
    for topping in toppings:
        print(f"- {topping}")


add('ham', 'cheese', 'tomato')
add('ham', 'cheese')
add('ham')