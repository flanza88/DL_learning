message_list = ['admin', 'tom', 'jack', 'lucy', 'lily']
sent_messages = list()

def show_messages(message):
    print(message)

def send_messages(message_list):
    while message_list:
        message = message_list.pop()
        show_messages(message)
        sent_messages.append(message)

send_messages(message_list)
print(message_list)
print(sent_messages)