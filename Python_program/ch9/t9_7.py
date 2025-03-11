class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def describe_user(self):
        print(f"first_name is {self.first_name.title()}, last_name is {self.last_name}")

    def greet_user(self):
        print(f"Hello, {self.first_name.title()} {self.last_name}")


class Admin(User):
    def __init__(self, first_name, last_name):
        super().__init__(first_name, last_name)
        self.privileges = ["can add post", "can delete post", "can ban user"]

    def show_privileges(self):
        print(f"privileges are {self.privileges}")


user1 = Admin("elon", "musk")
user1.show_privileges()
