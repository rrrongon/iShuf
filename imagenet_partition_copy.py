import os
import os.path
import sys
import math
import pprint
import random
import time
import argparse
import zipfile
from mpi4py import MPI
import shutil, json

MINI = "MINI"
_21K = "_21K"
RAND = "RAND"
ISHUF = "iSHUF"

argumentparser = argparse.ArgumentParser()

wnid_file = 'wnids.txt'
words_file = 'words.txt'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def copy_partition(PARTITION_DIR, TARGET_DIR, rank, size, comm):
    pass

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    f = open('/home/r.rongon/research/project_shuffle/customdatasampler_rand/config.json')
    configs =json.load(f)

    DATASET = configs["DATA_TYPE"]
    EXP_TYPE = configs["EXP_TYPE"]

    if DATASET == _21K:
        OUT_FOLDER = './imagenet_dataset/imagenet21k_resized'
        PARTITION_DIR = './imagenet_dataset/imagenet21k_resized'
        TARGET_DIR = './imagenet_dataset/imagenet21k_resized'
    elif DATASET == MINI:
        OUT_FOLDER = configs["mini"] 
        PARTITION_DIR = configs["mini"] 
        TARGET_DIR = configs["mini"] 

    zip_files = [file for file in os.listdir(PARTITION_DIR) if file.endswith('.zip')]
    for idx, zip_file in enumerate(zip_files):
        if idx % size == rank:
            dir_name = os.path.splitext(zip_file)[0]
            _dir_path = os.path.join(TARGET_DIR, dir_name)
            dir_path = os.path.join(_dir_path, 'train')
            os.makedirs(dir_path, exist_ok=True)

            # Copy the zip file to the directory
            zip_path = os.path.join(PARTITION_DIR, zip_file)
            dest_path = os.path.join(dir_path, zip_file)
            shutil.copy2(zip_path, dest_path)

            # Extract the zip file
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(dir_path)

            # Remove the zip file
            os.remove(dest_path)

            wnid_path = os.path.join(PARTITION_DIR, wnid_file)
            dest_wnid_path = os.path.join(_dir_path, wnid_file)
            shutil.copy2(wnid_path, dest_wnid_path)

            words_path = os.path.join(PARTITION_DIR, words_file)
            dest_words_path = os.path.join(_dir_path, words_file)
            shutil.copy2(words_path, dest_words_path)

            #val_source = os.path.join(PARTITION_DIR, 'val')
            #val_target = os.path.join(_dir_path, 'val')

            #shutil.copytree(val_source, val_target)
    comm.Barrier()

if __name__ == '__main__':
   main(argumentparser.parse_args())

