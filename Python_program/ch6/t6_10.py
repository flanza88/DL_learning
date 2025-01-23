favorite_num = {'musk': [8,88,888], 
          'mike': [7,77],
          'lucy': [50,80],
          'jhon': [77,66,55],
          'lily' : [8,9,10]}

for key, values in favorite_num.items():
    print(f"{key} likes:")
    for value in values:
        print(f"\t{value}")
