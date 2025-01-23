cities = {
    'beijing': {
        'population': 1000,
        'area': 100,
        'fact' : 'beijing is the capital of china'
    }, 
    'shanghai': {
        'population': 888,
        'area': 88,
        'fact' : 'shanghai is rich'
    }, 
    'guangzhou': {
        'population': 99,
        'area': 9,
        'fact' : 'guangzhou is hot'
    }, 
}
for city, city_info in cities.items():
    print(city)
    print(city_info)
