# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: read_data.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 20-1-13
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import tarfile
import cv2
import pickle
import h5py
import lmdb

import numpy as np
from PIL import Image


def read_tarfile(filepath, index):
    with tarfile.open(filepath, mode='r') as tar:
        data = tar.extractfile('{:04d}.pkl'.format(index))
        data = pickle.load(data)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        print('Type: {}, shape: {}'.format(type(data), data.shape))

        # img = Image.fromarray(data)
        # img = img.convert('RGB')
        # img.show()
        return data



def read_h5py(filepath, index):
    fw = h5py.File(filepath, mode='r')
    data = fw[str(index)][()]
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    print('Type: {}, shape: {}'.format(type(data), data.shape))

    # img = Image.fromarray(data)
    # img = img.convert('RGB')
    # img.show()
    return data


def read_lmdb(filepath, index):
    lmdb_env = lmdb.open(filepath)
    lmdb_txn = lmdb_env.begin()
    cursor = lmdb_txn.cursor()
    key_index = bytes(str(index), encoding='utf8')
    value = cursor.get(key_index)
    data = np.frombuffer(value)

    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    print('Type: {}, shape: {}'.format(type(data), data.shape))

    return data
    #
    # img = Image.fromarray(data)
    # img = img.convert('RGB')
    # img.show()


def main():
    filepath = 'images-data.tar'
    img0_tar = read_tarfile(filepath, 0)

    filepath = 'images-data.h5'
    img0_h5 = read_h5py(filepath, 0)

    filepath = 'images-data.lmdb'
    img0_lmdb = read_lmdb(filepath, 0)

    print('{:.4f}, {:.4f}'.format(np.sum(img0_h5 - img0_lmdb), np.sum(img0_tar - img0_lmdb)))


if __name__ == "__main__":
    main()


'''bash
python read_data.py
'''