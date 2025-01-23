def describe_pet( pet_name,animal_type='dog'):#确定的参数放后面
    print(f"\nI have a {animal_type}.")
    print(f"My {animal_type}'s name is {pet_name.title()}.")

describe_pet(animal_type='hamster',  pet_name='harry')#使用关键字参数
describe_pet('willie')#使用位置参数