from mpi4py import MPI
import sys
import os,json

from imagenet_customDatasetInterface import ImageNetDataset
from imagenet_nodeComm import ImageNetNodeCommunication

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dataset = list()
batch_size = 32
no_of_batch = 4
total = batch_size * no_of_batch

ROOT_DIR = './imagenet_dataset/imagenet-mini'

train_folder = os.path.join(ROOT_DIR,"parition" + str(rank)+"/train")
wnids_file = os.path.join(ROOT_DIR,"parition" + str(rank)+"/wnids.txt")
words_file = os.path.join(ROOT_DIR,"parition" + str(rank)+"/words.txt")

train_dataset = ImageNetDataset(train_folder, wnids_file, words_file, transform=None)
EPOCHS = 50
fraction = 0.3
seed = 42

nc = ImageNetNodeCommunication(train_dataset, batch_size, fraction, seed)

for epoch in range(EPOCHS):
    nc.scheduling(epoch)
    nc._communicate()
    print("Rank#{0} is done in Communicating".format(rank))
    sys.stdout.flush()
    comm.Barrier()

    nc.sync_recv()
    comm.Barrier()
    print("Rank#{0} is done in sync recv".format(rank))
    sys.stdout.flush()

    nc.sync_send()
    print("Rank#{0} is done in sync send".format(rank))
    sys.stdout.flush()
    comm.Barrier()

    nc.clean_sent_samples()
    print("Rank#{0} is done in clean sent samples".format(rank))
    sys.stdout.flush()
    comm.Barrier()
