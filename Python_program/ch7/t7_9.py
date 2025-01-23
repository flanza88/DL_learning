
sandwich_orders  = ['potato', 'pastrami','tomato', 'cheese','pastrami','pastrami']
finished_sandwiches = []
print("食店的五香烟熏牛肉（pastrami）卖完了")

while 'pastrami' in sandwich_orders:
    sandwich_orders.remove('pastrami')

for sandwich_order in sandwich_orders:
    print("I made your " + sandwich_order + " sandwich.")
    finished_sandwiches.append(sandwich_order)

print(f"I made these sandwiches:{finished_sandwiches}")