def make_album(singer='', album='',num = None):
    if num is None:
        return {'singer': singer, 'album': album}
    else:
        return {'singer': singer, 'album': album, 'num' : num}

print(make_album('taylor swift','lover',20))
print(make_album('jay chou','jay'))