import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import os, json
import pandas as pd
from myDataSampler import CustomSampler
from utility.sampler import DistributedSampler as dsampler
import horovod.torch as hvd
from  torchvision import transforms, models
import PIL
from mpi4py import MPI
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import time
import torch.utils.data.distributed as DS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn as nn

#File Import
from imagenet_customDatasetInterface import ImageNetDataset
from imagenet_nodeComm import ImageNetNodeCommunication
from imagenetValidation import ImageNetValidationDataset

MINI = 1
_21K = 2
DATASET = _21K # /21K

if DATASET == _21K:
    OUT_FOLDER = './imagenet_dataset/imagenet21k_resized'
    PARTITION_DIR = './imagenet_dataset/imagenet21k_resized'
    TARGET_DIR = './imagenet_dataset/imagenet21k_resized'
    CLASS_NUMBER = 21844
elif DATASET == MINI:
    OUT_FOLDER = './imagenet_dataset/imagenet-mini'
    PARTITION_DIR = './imagenet_dataset/imagenet-mini'
    TARGET_DIR = './imagenet_dataset/imagenet-mini'
    CLASS_NUMBER = 1000

if __name__ == '__main__':

    _is_cuda = torch.cuda.is_available()
    if not _is_cuda:
        print("There is no CUDA available\n")
        exit(1)

    hvd.init()
    rank = hvd.rank()
    f = open('config.json')
    configs =json.load(f)
    torch.manual_seed(configs["MODEL"]["seed"])

    if DATASET == MINI:
        IMGNET_DIR = configs["ROOT_DATADIR"]["imgnet_dir"]
    elif DATASET == _21K:
        IMGNET_DIR = configs["ROOT_DATADIR"]["imgnet_21k_dir"]

    train_folder = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/train")
    wnids_file = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/wnids.txt")
    words_file = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/words.txt")
    val_wnid_file = os.path.join(IMGNET_DIR,"ImageNet_val_label.txt")
    class_label_file = os.path.join(IMGNET_DIR,"class-label.txt")
    _val_folder = "./imagenet_dataset/imagenet21k_resized/val"
    val_wnid_file = "./imagenet_dataset/imagenet21k_resized/ImageNet_val_label.txt"


    train_dataset = ImageNetDataset(train_folder, wnids_file, words_file,class_label_file, CLASS_NUMBER, transform=None)

    train_dataset_len = len(train_dataset)
    # Keep the minimum training dataset length for being in sync
    train_dataset_len = torch.tensor(train_dataset_len)
    min_train_dataset_len = hvd.allreduce(train_dataset_len, op=hvd.mpi_ops.Min)
    min_train_dataset_len = min_train_dataset_len.item()

    _no_of_intraNode_workers = hvd.size() #configs["MODEL"]["no_of_batches"]
    _batch_size = configs["MODEL"]["batch_size"]


    mini_batch_limit = (min_train_dataset_len / _batch_size)
    print("Rank#{0}: minimum batch number limit#{1}".format(rank,mini_batch_limit))

    #custom_sampler = CustomSampler(_train_dataset)
    _train_sampler = dsampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
    _train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=_batch_size,
            sampler= _train_sampler)

    #_val_folder = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/val")
    #_val_folder = os.path.join(IMGNET_DIR, "val")

    #_val_dataset = ImageNetDataset(_val_folder, wnids_file, words_file, CLASS_NUMBER, transform=None)
    _val_dataset = ImageNetValidationDataset(_val_folder, val_wnid_file, class_label_file, CLASS_NUMBER, transform=None)
    #custom_sampler = CustomSampler(_val_dataset)
    _val_sampler = dsampler(
            _val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle = True)
    _val_loader = torch.utils.data.DataLoader(
            _val_dataset, batch_size= _batch_size,
            sampler= _val_sampler)

    print("training directory of rank {0} is {1}. training set len {2} and val set len{3}".format(rank, train_folder , len(train_dataset), len(_val_dataset)))

    wd = configs["MODEL"]["wd"]
    use_adasum = 0
    MPI.COMM_WORLD.Barrier()

    model = models.resnet50(pretrained=True)

    num_classes = CLASS_NUMBER
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    base_lr = configs["MODEL"]["base_lr"]
    momentum = configs["MODEL"]["moment"]

    scaled_lr = base_lr * hvd.size()
    if _is_cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        #if use_adasum and hvd.nccl_built():
        #    lr_scaler = configs["MODEL"]["batch_size"] * hvd.local_size()

    MPI.COMM_WORLD.Barrier()
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr,
                          momentum=momentum, weight_decay=wd)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    loss_fn = nn.CrossEntropyLoss()

    MPI.COMM_WORLD.Barrier()
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    batch_size = configs["MODEL"]["batch_size"]
    fraction = configs["MODEL"]["fraction"]
    seed = configs["MODEL"]["seed"]

    epoch_no = configs["MODEL"]["epoch"]

    total_duration = 0

    nc = ImageNetNodeCommunication(train_dataset, batch_size, fraction, seed, min_train_dataset_len, epoch_no)

    for epoch in range(epoch_no):
        nc.scheduling(epoch)
        nc._communicate()
        print("Rank#{0} is done in Communicating".format(rank))
        sys.stdout.flush()
        torch.cuda.synchronize()
        hvd.allreduce(torch.tensor(0), name="barrier")

        nc.sync_recv()
        print("Rank#{0} is done in sync recv".format(rank))
        sys.stdout.flush()
        torch.cuda.synchronize()
        hvd.allreduce(torch.tensor(0), name="barrier")

        nc.sync_send()
        print("Rank#{0} is done in sync send".format(rank))
        sys.stdout.flush()
        torch.cuda.synchronize()
        hvd.allreduce(torch.tensor(0), name="barrier")

        nc.clean_sent_samples()
        print("Rank#{0} is done in clean sent samples".format(rank))
        sys.stdout.flush()
        torch.cuda.synchronize()
        hvd.allreduce(torch.tensor(0), name="barrier")

