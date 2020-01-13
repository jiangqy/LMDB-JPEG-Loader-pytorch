# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: compress_data.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 20-1-13
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
filelists = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']

import os
import tarfile
import pickle
import cv2
import io
import h5py
import lmdb
import sys

import numpy as np

from PIL import Image


def write_to_tarfile(images):
    with tarfile.open('images-data.tar', mode='w') as tar:
        for index, image in enumerate(images):
            info = tarfile.TarInfo('{:04d}.pkl'.format(index))
            img = Image.open(os.path.join('images', image))
            img = img.convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            data = np.array(img)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, img = cv2.imencode('.jpg', data, encode_param)
            data = pickle.dumps(img)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        print('save images done')


def write_to_h5py(images):
    fw = h5py.File('images-data.h5', mode='w')
    for index, image in enumerate(images):
        img = Image.open(os.path.join('images', image))
        img = img.convert('RGB')
        img = img.resize((256, 256), Image.BILINEAR)
        data = np.array(img)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img = cv2.imencode('.jpg', data, encode_param)
        fw.create_dataset(name=str(index), data=img)
    fw.close()


def write_to_lmdb(images):
    lmdb_env = lmdb.open('images-data.lmdb', map_size=10240000)
    with lmdb_env.begin(write=True) as lmdb_txn:
        for index, image in enumerate(images):
            img = Image.open(os.path.join('images', image))
            img = img.convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            data = np.array(img)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, img = cv2.imencode('.jpg', data, encode_param)
            print(type(img))
            lmdb_txn.put(str(index).encode(), img.tobytes())


def main():
    write_to_tarfile(filelists)
    write_to_h5py(filelists)
    write_to_lmdb(filelists)


if __name__ == "__main__":
    main()

'''bash
python compress_data.py
'''
