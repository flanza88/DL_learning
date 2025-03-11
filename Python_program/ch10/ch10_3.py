try:
    print(5 / 0)
except ZeroDivisionError:
    print("You can't divide by zero!")

print("Give me two numbers, and I'll divide them.")
print("Enter 'q' to quit.")
while True:
    first_number = input("\nFirst number: ")
    if first_number == "q":
        break
    second_number = input("Second number: ")
    if second_number == "q":
        break
    try:
        answer = int(first_number) / int(second_number)
    except ZeroDivisionError:
        print("You can't divide by 0!")
    else:  # 仅在try代码块成功执行时才需要运行的代码，这些代码应放在else 代码块中
        print(answer)


def count_words(filename):
    try:
        with open(filename, encoding="utf-8") as f:
            contents = f.read()
    except FileNotFoundError:
        print(f"Sorry, the file {filename} does not exist.")
    else:
        words = contents.split()
        num_words = len(words)
        print(f"The file {filename} has about {num_words} words.")


filename = "alice.txt"
count_words(filename)

filenames = ["alice.txt", "siddhartha.txt", "moby_dict.txt", "little_women.txt"]
for filename in filenames:
    count_words(filename)
