# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: data_loader.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 20-1-13
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import h5py
import cv2
import pickle

import torch.utils.data as data

from torch.utils.data import DataLoader


class DataSet(data.Dataset):
    def __init__(self):
        filepath = 'images-data.h5'
        self.file_handler = h5py.File(filepath, mode='r')
        self.num_images = len(self.file_handler.keys())

    def __getitem__(self, item):
        data = self.file_handler[str(item)][()]
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data, item

    def __len__(self):
        return self.num_images


def create_data_loader():
    dset = DataSet()
    loader = DataLoader(dset,
                        batch_size=2,
                        shuffle=False,
                        num_workers=2)

    for idx, (images, index) in enumerate(loader):
        print(images.shape)


def main():
    create_data_loader()


if __name__ == "__main__":
    main()

'''bash
python data_loader.py
'''
