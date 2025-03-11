import random


class Die:
    def __init__(self, sides=6):
        self.sides = sides

    def roll(self):
        return random.randint(1, self.sides)


die6 = Die(6)
die10 = Die(10)
die20 = Die(20)
for i in range(10):
    print(die6.roll(), die10.roll(), die20.roll())
