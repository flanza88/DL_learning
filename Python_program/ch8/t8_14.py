def make_car(maker, model, **car_info):
    car_info['maker'] = maker
    car_info['model'] = model
    return car_info
car = make_car('subaru', 'outback',color='blue',tow_package=True)
print(car)
