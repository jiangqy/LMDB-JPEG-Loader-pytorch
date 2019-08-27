# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: write_read_lmdb.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 19-8-27
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import lmdb
import cv2

import numpy as np


def readjpeg(value):
    X = cv2.imdecode(value, cv2.IMREAD_COLOR)
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    return X


def get_imagelists():
    imagelists = []
    with open('images.list', 'r') as fp:
        for lines in fp:
            tmps = lines.strip().split(' ')
            imagelists.append(tmps)
    return imagelists


def write_to_lmdb():
    imagelists = get_imagelists()
    lmdb_path = 'data/image-lmdb.db'

    env = lmdb.open(lmdb_path, map_size=1e9)
    txn = env.begin(write=True)

    for idx, tmps in enumerate(imagelists):
        image, imagepath = tmps[0], tmps[1]
        with open(imagepath, 'rb') as f:
            jpeg = f.read()
        jpeg = np.asarray(bytearray(jpeg), dtype='uint8').tobytes()
        txn.put(key=str(idx).encode(), value=jpeg)
    txn.commit()
    env.close()


def read_from_lmdb():
    images = []
    env_db = lmdb.Environment('data/image-lmdb.db')
    txn = env_db.begin()
    cur = txn.cursor()
    key_index = bytes('0', encoding='utf8')
    value = cur.get(key_index)
    jpeg = np.frombuffer(value, dtype='uint8')
    image = readjpeg(jpeg)
    print(image.shape)
    from PIL import Image
    image = Image.fromarray(image)
    image.show()
    env_db.close()


def main():
    write_to_lmdb()
    read_from_lmdb()


if __name__ == "__main__":
    main()


