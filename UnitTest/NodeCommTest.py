from mpi4py import MPI
import sys
sys.path.append('..')
from InterNodeCommunication import NodeCommunication

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

dataset = list()
batch_size = 32
no_of_batch = 4
total = batch_size * no_of_batch


for i in range(0,batch_size):
    dataset.append(i+(rank*batch_size))

if rank == 1:
    for i in range(batch_size):
        assert(dataset[i] == (rank*batch_size)+i), "Error data insertion in rank {rank}"

fraction = 0.2
seed = 10
print("proc: {0}".format(rank))
nc = NodeCommunication(dataset,batch_size, fraction, seed)
nc._communicate(batch_index=5)
