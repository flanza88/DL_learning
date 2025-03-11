class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        self.number_served = 0

    def describe_restaurant(self):
        print(f"restaurant_name is {self.restaurant_name}")
        print(f"cuisine_type is {self.cuisine_type}")

    def open_restaurant(self):
        print("Restaurant is opening")

    def set_number_served(self, number):
        self.number_served = number

    def increment_number_served(self, increment_number):
        self.number_served += increment_number


restaurant = Restaurant("eat_shit", "shit")
restaurant.describe_restaurant()
restaurant.open_restaurant()

restaurant.set_number_served(10)
print(f"number_served = {restaurant.number_served}")
restaurant.increment_number_served(100)
print(f"number_served = {restaurant.number_served}")
