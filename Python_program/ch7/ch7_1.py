message = input("Tell me something, and I will repeat it back to you: ")
print(message)
name = input("Please enter your name: ")
print(f"\nHello, {name}!")

height = int(input("How tall are you, in inches? "))
if height >= 48:
    print("\nYou're tall enough to ride!")
else:
    print("\nYou'll be able to ride when you're a little older.")