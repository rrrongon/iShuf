import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed

import horovod.torch as hvd

import os
import zipfile
import os.path
import re
import math
import time
import sys
from mpi4py import MPI

import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

from utility.partition_download_localnode import download_toLocal

if __name__ == '__main__':

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    print("rank {0} and size{1}".format(rank, size))
    #train_folder = "/home/rongon/Documents/research/shuffling/Codes/ProjectCode/natural_image/Partition_Folder"
    #target_local_train_folder = "/home/rongon/Documents/research/shuffling/Codes/ProjectCode/natural_image/Partition_Folder"
    #download_toLocal(train_folder, target_local_train_folder, rank, size)
