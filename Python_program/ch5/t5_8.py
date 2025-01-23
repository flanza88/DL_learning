user_list = ['admin', 'john', 'lucy', 'jack', 'musk']
for user in user_list:
    if user == 'admin':
        print("Hello admin, would you like to see a status report?")
    else:
        print("Hello " + user + ", thank you for logging in again.")
