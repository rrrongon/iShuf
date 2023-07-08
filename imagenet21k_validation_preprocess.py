from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

val_path = './imagenet_dataset/imagenet21k_resized/val'

def process_folder(folder):
    folder_path = os.path.join(val_path, folder)
    files = os.listdir(folder_path)

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        sample_index = i + 1

        if sample_index % 2 != 0:
            os.remove(file_path)

if rank == 0:
    folders = os.listdir(val_path)
    folders_per_process = len(folders) // size
    remainder = len(folders) % size

    for i in range(size-1):
        start = i * folders_per_process
        end = start + folders_per_process
        if i == size - 1:
            end += remainder
        comm.send(folders[start:end], dest=i + 1, tag=1)

for i in range(1, size + 1):
    if rank == i:
        folders = comm.recv(source=0, tag=1)
        for folder in folders:
            process_folder(folder)

MPI.Finalize()

