def build_profile(first, last, **user_info):
    user_info['first_name'] = first
    user_info['last_name'] = last
    return user_info
user_profile = build_profile('albert', 'einstein',location='princeton',field='physics')
print(user_profile)

my_profile = build_profile('flanza', 'fucker',location='china',field='nan', age=18)
print(my_profile)