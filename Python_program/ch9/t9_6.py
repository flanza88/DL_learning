class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type

    def describe_restaurant(self):
        print(f"restaurant_name is {self.restaurant_name}")
        print(f"cuisine_type is {self.cuisine_type}")

    def open_restaurant(self):
        print("Restaurant is opening")


class IceCreamStand(Restaurant):
    def __init__(self, restaurant_name, cuisine_type, flavors):
        super().__init__(restaurant_name, cuisine_type)
        self.flavors = flavors


# restaurant = IceCreamStand("eat_shit", "shit", ["tomato", "potato", "milk"])
# print(f"flavors:{restaurant.flavors}")
