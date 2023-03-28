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

    # Create subfolder for each rank
    local_folder = os.path.join(local_target_folder,str(rank))

    command = "rm -r " + str(local_folder)
    os.system(command)
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    # Copy partitions from root to local
    source_file_name = os.path.join(source_folder, "class_to_idx.txt")
    dest_file_name = os.path.join(local_folder, "class_to_idx.txt")
    command = "cp " + str(source_file_name) + " " + str(dest_file_name)
    os.system(command)

    for i in range(0, len(rank_partition_files)):
        zip_file_name = rank_partition_files[i]
        source_file_name = os.path.join(source_folder, zip_file_name)
        local_file_name = os.path.join(local_folder, zip_file_name)
        command = "cp " + str(source_file_name) + " " + str(local_folder)
        #print(command)
        os.system(command)

        with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
            zip_ref.extractall(local_folder)

        command = "rm " + str(local_file_name)
        #print(command)
        os.system(command)

    hvd.barrier()
    print("Complete coping from rank {rank}")
