# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: create_loader.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 19-8-27
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import cv2
import lmdb

import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader


class LMDBDataset(data.Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True)
        self.txn = self.env.begin()

        self.length = self.env.stat()['entries'] - 1

    @staticmethod
    def __decodejpeg(jpeg):
        x = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def __getitem__(self, item):
        cursor = self.txn.cursor()
        key_index = bytes('0', encoding='utf8')
        value = cursor.get(key_index)
        jpeg = np.frombuffer(value, dtype='uint8')
        image = self.__decodejpeg(jpeg)
        return image, item

    def __len__(self):
        return self.length


def create_dataset(lmdb_path):
    dset = LMDBDataset(lmdb_path)
    return dset


def create_loader(lmdb_path, batch_size, shuffle=False):
    dset = LMDBDataset(lmdb_path)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    return loader
