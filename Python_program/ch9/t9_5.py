class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
        self.login_attempts = 0

    def describe_user(self):
        print(f"first_name is {self.first_name.title()}, last_name is {self.last_name}")

    def greet_user(self):
        print(f"Hello, {self.first_name.title()} {self.last_name}")

    def increment_login_attempts(self):
        self.login_attempts += 1

    def reset_login_attempts(self):
        self.login_attempts = 0


user1 = User("elon", "musk")
user2 = User("justin", "bieber")
user1.describe_user()
user2.greet_user()

user1.increment_login_attempts()
user1.increment_login_attempts()
user1.increment_login_attempts()
print(f"login_attempts is {user1.login_attempts}")
user1.reset_login_attempts()
print(f"login_attempts is {user1.login_attempts}")
