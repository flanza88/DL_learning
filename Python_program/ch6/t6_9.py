favorite_places = {'musk': ['beijing'], 
          'mike': ['shanghai','guangzhou','shenzhen'],
          'john': ['usa', 'japan']}


for name,places in favorite_places.items():
    print(f"{name} likes :")
    for place in places:
        print(f"\t{place}")