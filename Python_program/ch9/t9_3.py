class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def describe_user(self):
        print(f"first_name is {self.first_name.title()}, last_name is {self.last_name}")

    def greet_user(self):
        print(f"Hello, {self.first_name.title()} {self.last_name}")


user1 = User("elon", "musk")
user2 = User("justin", "bieber")
user1.describe_user()
user2.greet_user()
