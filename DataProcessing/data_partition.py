import torch

import numpy as np
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

argumentparser = argparse.ArgumentParser()
argumentparser.add_argument('-f','--train-folders-path', help='Folder path of datasets', required=True)
argumentparser.add_argument('-np','--npartitions', help='<Required> Number of partitions', required=True)
argumentparser.add_argument('-nf','--nfile', default=1024, help='Number of files per chunks. Each partitions may include many chunks')
argumentparser.add_argument('-s','--shuffle', default=True, help='Shuffle before partitioning or not?')
argumentparser.add_argument('-dl','--droplast', default=True, help='(bool, optional): if True, then the sampler will drop the tail of the data to make it evenly divisible across the number of partitions. If False, the sampler will add extra indices to make the data evenly divisible across the partitions')
argumentparser.add_argument('-o', '--out', help="output folder")
argumentparser.add_argument('-cf', '--class-file', help="class to idx file name")

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_classFile(filename):
    classes = []
    indices = []
    with open(filename, 'r') as f:
        for line in f:
            class_name, index = line.strip().split()
            classes.append(class_name)
            indices.append(int(index))

    class_to_idx = {name: int(number) for name, number in zip(classes, indices)}

    return classes, class_to_idx

#dir = train folder
def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if is_image_file(filename):
                path = '{0}/{1}'.format(target, filename) #path = "class/filename"
                item = (path, class_to_idx[target])
                images.append(item)

    return images

def main(args):
    # 1. Argument checking
    print ("Run with arguments:")
    for key, value in args._get_kwargs():
        if value is not None:
            print(value,key)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    train_root = os.path.abspath(args.train_folders_path)
    num_partitions = int(args.npartitions)
    file_per_chunk = int(args.nfile)
    is_shuffle = args.shuffle
    drop_last = args.droplast
    out_folder = os.path.abspath(args.out)
    class_file = os.path.abspath(args.class_file)

    classes, class_to_idx = read_classFile(class_file)
    imgs = make_dataset(train_root, class_to_idx)

    if (drop_last) and len(imgs) % num_partitions != 0:
        num_samples = math.ceil((len(imgs) - num_partitions) / num_partitions)
    else:
        num_samples = math.ceil(len(imgs) / num_partitions)
    total_size = num_partitions * num_samples

    # 3. Shuffle before partitioning
    if is_shuffle == True:
        g = torch.Generator()
        g.manual_seed(42)
        indices = torch.randperm(len(imgs), generator=g).tolist()
    else:
        indices = list(range(len(imgs)))

    # remove tail of data to make it evenly divisible.
    indices = indices[:total_size]
    assert len(indices) == total_size , "indices is not equal to total size. indices size {0}!= totalsize{1}".format(len(indices), total_size)

    partition_indices = [None] * num_partitions
    for i in range(0, num_partitions):
        partition_indices[i] = indices[i:total_size:num_partitions]

    # 6. Save class_to_idx file
    if rank ==0:
        out_file_name = os.path.join(out_folder,"class_to_idx.txt")
        f = open(out_file_name, "w")
        for class_name, idx in class_to_idx.items():
            f.write(str(class_name) + "\t" + str(idx) + "\n")
        f.close()
        print("Write into: ",out_file_name)
    

    for i in range(0, num_partitions):
        if i% size == rank:
            try:
                print("Rank", rank, "Process", i, "over", num_partitions, "partitions")
                zip_filename = os.path.join(out_folder,"parition" + str(i) + ".zip")
                zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
                for new_idx in partition_indices[i]:
                    path, class_name = imgs[new_idx]
                    split_path = os.path.splitext(path)
                    arc_path = split_path[0] + "_1" + split_path[1]
                    print("Rank", rank, "Save", new_idx, "with", arc_path)

                    zipf.write(os.path.join(train_root, path), arcname=arc_path)
                zipf.close()
            except Exception as e:
                print("Error {0}".format(e))
                
if __name__ == '__main__':
   main(argumentparser.parse_args())  