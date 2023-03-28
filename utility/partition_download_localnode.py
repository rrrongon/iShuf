'''
Goal of this code is to copy partitions of codes to local Node's training folders of each rank.
In large scale each compute node will copy partitions to that node's local storage.
Those folder of data would be the only training sample of the Ranks.
So, actions of this code is
- Copy training data partition to another foler (Later folder would be node's local storage)
- Unzip and create the dataset for the Node (right now create dataset by different folder and rank. So that each rank have different portion of data set.)
- Return the dataset to code
'''

import torch

import numpy as np
import os
import os.path
import re
import sys

import time
import zipfile
import horovod.torch as hvd

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
ZIP_EXTENSIONS = [".zip"]


def is_zip_file(filename):
    return any(filename.endswith(extension) for extension in ZIP_EXTENSIONS)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def download_toLocal(train_folder, target_local_train_folder, rank, size):
    source_folder = os.path.abspath(train_folder)
    local_target_folder = os.path.abspath(target_local_train_folder)

    partition_files = list() # different for each rank
    for partition_filename in sorted_alphanumeric(os.listdir(source_folder)): #.sort():
        if is_zip_file(partition_filename):
            partition_files.append(partition_filename)
    num_partitions = len(partition_files)
    if rank == 0:
        print(num_partitions,partition_files)
    
    rank_partition_files = list()
    for i in range(0, num_partitions):
        if i% size == rank:
            rank_partition_files.append(partition_files[i])
    
    print("Rank ", rank, " will download " , len(rank_partition_files), " number of foles. Partitions are :", rank_partition_files)