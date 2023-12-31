import os, json
import os.path
import sys
import math
import pprint
import random
import time
import argparse
import zipfile
from mpi4py import MPI
import shutil

MINI = "MINI"
_21K = "_21K"
RAND = "RAND"
ISHUF = "iSHUF"

'''
if DATASET == _21K:
    OUT_FOLDER = './imagenet_dataset/imagenet21k_resized'
    PARTITION_DIR = './imagenet_dataset/imagenet21k_resized'
    TARGET_DIR = './imagenet_dataset/imagenet21k_resized'
elif DATASET == MINI:
    OUT_FOLDER = './imagenet_dataset/imagenet-mini'
    PARTITION_DIR = './imagenet_dataset/imagenet-mini'
    TARGET_DIR = './imagenet_dataset/imagenet-mini'
'''

argumentparser = argparse.ArgumentParser()
#argumentparser.add_argument('-npp','--npp', help='<Required> Number of Nodes to partition data among', required=True)
#argumentparser.add_argument('-f','--root-dir', help='Folder path of datasets', required=True)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def copy_partition(PARTITION_DIR, TARGET_DIR):
    zip_files = [file for file in os.listdir(PARTITION_DIR) if file.endswith('.zip')]
    for idx, zip_file in enumerate(zip_files):
        dir_name = os.path.splitext(zip_file)[0]
        dir_path = os.path.join(TARGET_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        # Copy the zip file to the directory
        zip_path = os.path.join(PARTITION_DIR, zip_file)
        dest_path = os.path.join(dir_path, zip_file)
        shutil.copy2(zip_path, dest_path)

        # Extract the zip file
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(dir_path)

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

    #np = int(args.npp) # Consider that each process will work for each node to partition training data
    #root_dir = os.path.abspath(args.root_dir) #training directory. we should get a lot of wnid folders

    np = 4
    #root_dir = '/scratch/user/r.rongon/myprefix_20230921_182522/imagenet_dataset/imagenet-mini/train' #os.path.abspath(args.root_dir) #training directory. we should get a lot of wnid folders
    root_dir = PARTITION_DIR + '/train'

    # Get all directories in the directory
    wnids = [d for d in os.listdir(root_dir) if os.path.isdir(root_dir)]
    # Print the directory names

    zip_files = list()

    for wnid in wnids:
        # Get all files in the directory
        wnid_dir = os.path.join(root_dir, wnid)
        files = [f for f in os.listdir(wnid_dir) if os.path.isfile(os.path.join(wnid_dir, f))]

        for index, file_path in enumerate(files):
            if index % size == rank:
                file_path = os.path.join(wnid_dir, file_path)
                partition_no = index % np

                # add the file to proper partition zip file
                # figure out the proper zip file name
                zip_files.append((partition_no, file_path))
                #print("File path#{0} added to Partition no#{1} ".format(file_path, partition_no))

        comm.Barrier()
    comm.Barrier()

    partition_pair = dict()
    counter = 0
    for each_pair in zip_files:

        if DATASET == _21K:
            #if counter % 2 ==0:
            partition_no = each_pair[0]
            sample_path = each_pair[1]

            if partition_no in partition_pair:
                partition_pair[partition_no].append(sample_path)
            else:
                partition_pair[partition_no] = [sample_path]
            counter += 1

        elif DATASET == MINI:
            partition_no = each_pair[0]
            sample_path = each_pair[1]

            if partition_no in partition_pair:
                partition_pair[partition_no].append(sample_path)
            else:
                partition_pair[partition_no] = [sample_path]


    if rank == 0:
        for partition_no in range(np):
            zip_filename = os.path.join(OUT_FOLDER,"parition" + str(partition_no) + ".zip")
            zip_directory = os.path.dirname(zip_filename)

            if not os.path.exists(zip_directory):
                os.makedirs(zip_directory)

            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                print(f" {zip_filename} created")
            zipf.close()

    comm.Barrier()

    for partition_no, paths in partition_pair.items():
        zip_filename = os.path.join(OUT_FOLDER,"parition" + str(partition_no) + ".zip")
        #zip_directory = os.path.dirname(zip_filename)

        #if not os.path.exists(zip_directory):
        #    os.makedirs(zip_directory)

        mode = 'a' if os.path.exists(os.path.dirname(zip_filename)) else 'w'
        with zipfile.ZipFile(zip_filename, mode, zipfile.ZIP_DEFLATED) as zipf:
            for file_path in paths:
                wnid = os.path.basename(os.path.dirname(file_path))
                wnid_dir = os.path.join(root_dir, wnid)
                arc_path = os.path.relpath(file_path, wnid_dir)
                arc_path = wnid+'/'+arc_path
                zipf.write(file_path, arcname=arc_path)
                #print("{0} file adding to ZIP #{1}".format(file_path, zip_filename))
        zipf.close()

        print("Rank# {0} closed the partition# {1}.".format(rank, partition_no))

    #copy_partition(PARTITION_DIR, TARGET_DIR)

if __name__ == '__main__':
   main(argumentparser.parse_args())

