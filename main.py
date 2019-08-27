# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: main.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 19-8-27
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from utils.write_to_lmdb import write_to_lmdb
from utils.create_loader import create_loader

def get_imagelists():
    imagelists = []
    with open('images.list', 'r') as fp:
        for lines in fp:
            tmps = lines.strip().split(' ')
            imagelists.append(tmps)
    return imagelists


def main():
    lmdb_path = 'data/image-lmdb.db'
    imagelists = get_imagelists()
    write_to_lmdb(lmdb_path, imagelists)
    loader = create_loader(lmdb_path, batch_size=2, shuffle=False)
    for idx, batch in enumerate(loader):
        image, index = batch
        print('#batch: {:3d}, batch size: {}'.format(idx, image.size()))


if __name__ == "__main__":
    main()


