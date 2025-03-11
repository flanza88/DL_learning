class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type

    def describe_restaurant(self):
        print(f"restaurant_name is {self.restaurant_name}")
        print(f"cuisine_type is {self.cuisine_type}")

    def open_restaurant(self):
        print("Restaurant is opening")


restaurant = Restaurant("eat_shit", "shit")
restaurant.describe_restaurant()
restaurant.open_restaurant()
