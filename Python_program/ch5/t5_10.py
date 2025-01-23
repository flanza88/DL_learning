current_users = ['user0', 'user1', 'user2', 'user3', 'user4']
new_users = ['user0', 'user5', 'user2', 'user7', 'user8']

for user in new_users:
    if user in current_users:
        print(f"{user}需要输入别的用户名")
    else:
        print(f"{user}这个用户名未被使用")

