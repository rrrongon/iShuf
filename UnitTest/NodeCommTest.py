from mpi4py import MPI
import sys
import os,json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

from InterNodeCommunication import NodeCommunication
from dataloader_test import CustomDataset
from mpi4py import MPI
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = "hello"
    comm.send(data, dest=1)
elif rank == 1:
    data = comm.recv(source=0)
    print(data)

'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

f = open('config.json')
configs =json.load(f)

dataset = list()
batch_size = 32
no_of_batch = 4
total = batch_size * no_of_batch


for i in range(0,batch_size):
    dataset.append(i+(rank*batch_size))

if rank == 1:
    for i in range(batch_size):
        assert(dataset[i] == (rank*batch_size)+i), "Error data insertion in rank {rank}"

#root_dir = configs["ROOT_DATADIR"]["train_dir"]
root_dir = configs["ROOT_DATADIR"]["partiion_dir"]
root_dir = root_dir + str(rank) + "_data_2/"
_train_dir = root_dir + "train/"
_label_file_path = os.path.join(root_dir, "train_filepath.csv")
_train_dataset = CustomDataset(img_dir=_train_dir, label_file_path=_label_file_path)

EPOCHS = 50
fraction = 0.3
seed = 10
print("proc: {0}".format(rank))
nc = NodeCommunication(_train_dataset, batch_size, fraction, seed)

for epoch in range(EPOCHS):
    nc.scheduling(epoch)
    nc._communicate(epoch)
    print("Rank#{0} is done in Communicating".format(rank))
    sys.stdout.flush()
    comm.Barrier()

    nc.sync_recv()
    comm.Barrier()

    #if rank ==0:
    #    comm.Abort()
    #nc.sync_recv()
    nc.sync_send()
    comm.Barrier()

    nc.clean_sent_samples()
    comm.Barrier()
