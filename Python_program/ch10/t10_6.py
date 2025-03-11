a = input("Enter first number: ")
b = input("Enter second number: ")
try:
    a = int(a)
    b = int(b)
except ValueError:
    print("Please enter a number")
else:
    print(a + b)
