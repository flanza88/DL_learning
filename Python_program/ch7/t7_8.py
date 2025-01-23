
sandwich_orders  = ['potato', 'tomato', 'cheese']
finished_sandwiches = []
for sandwich_order in sandwich_orders:
    print("I made your " + sandwich_order + " sandwich.")
    finished_sandwiches.append(sandwich_order)

print(f"I made these sandwiches:{finished_sandwiches}")