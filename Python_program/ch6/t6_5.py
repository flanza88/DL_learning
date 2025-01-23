dictionary = {'nile': 'egypt', 
          'river1': 'country1',
          'river2': 'country2',}

for key, value in dictionary.items():
    print(f"The {key} runs through {value}")

for key in dictionary.keys():
    print(key)
for value in dictionary.values():
    print(value)
