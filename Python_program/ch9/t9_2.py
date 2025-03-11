class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type

    def describe_restaurant(self):
        print(f"restaurant_name is {self.restaurant_name}")
        print(f"cuisine_type is {self.cuisine_type}")

    def open_restaurant(self):
        print("Restaurant is opening")


restaurant1 = Restaurant("eat_shit", "shit")
restaurant2 = Restaurant("eat_ass", "ass")
restaurant3 = Restaurant("eat_dick", "dick")
restaurant1.describe_restaurant()
restaurant2.describe_restaurant()
restaurant3.describe_restaurant()
