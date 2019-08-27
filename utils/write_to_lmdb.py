# -*- coding: UTF-8 -*-
#!/usr/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @FILE NAME: write_to_lmdb.py
# @AUTHOR: Jiang. QY.
# @MAIL: qyjiang24 AT gmail.com
# @DATE: 19-8-27
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import lmdb

import numpy as np


def write_to_lmdb(lmdb_path, imagelists, map_size=1e9):
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)

    for idx, tmps in enumerate(imagelists):
        image, imagepath = tmps[0], tmps[1]
        with open(imagepath, 'rb') as f:
            jpeg = f.read()
        jpeg = np.asarray(bytearray(jpeg), dtype='uint8').tobytes()
        txn.put(key=str(idx).encode(), value=jpeg)
    txn.commit()
    env.close()

