def make_album(singer='', album='',num = None):
    if num is None:
        return {'singer': singer, 'album': album}
    else:
        return {'singer': singer, 'album': album, 'num' : num}

print(make_album('taylor swift','lover',20))
print(make_album('jay chou','jay'))

while True:
    repeat = input("input q to quit or other to continue")
    if repeat == 'q':
        break
    singer = input("Please input singer's name:")
    album = input("Please input singer's album:")
    print(make_album(singer,album))