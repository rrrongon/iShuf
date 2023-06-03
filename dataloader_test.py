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

def get_accuracy(output, target):
    # get the index of the max log-probability
    ##print("Output tensor#{0}, output shape#{1} \n target tensor#{2}, target shape#{3}".format(output, output.size(),target,target.size()))
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

def custom_accuracy(output, target):
    row_max, pred_column_max = torch.max(output.data, 1)
    row_max, target_column_max = torch.max(target.data, 1)

    #print("column max#{0}\n target column max#{1}".format(pred_column_max, target_column_max))
    correct = (pred_column_max == target_column_max).sum().item()
    total = target.size(0)
    accuracy = correct / total

    #print("column max#{0}\n correct#{1}\n accuracy#{2}".format(pred_column_max, correct, accuracy))
    return accuracy * 100


class Result(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

    def avg(self):
        return self.sum/self.n

def plot_timeBreakdown(epochs, plt_comp_time, plt_reading_time, plt_shuffling_time):
    # Define the data for the stacked bar chart
    plt_total_time = np.add(np.add(plt_comp_time, plt_reading_time), plt_shuffling_time)
    barWidth = 0.85

    # Create the stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(epochs), plt_reading_time, color='#5DA5DA', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_shuffling_time, bottom=plt_reading_time, color='#FAA43A', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_comp_time, bottom=np.add(plt_reading_time, plt_shuffling_time), color='#60BD68', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_time, color='grey', alpha=0.2, edgecolor='white', width=barWidth)

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.xticks(range(epochs))
    plt.ylabel('Time (seconds)')
    plt.title('Training Time breakdown per epoch: PARTIAL')
    plt.legend(['Reading time', 'Shuffling time', 'Computation time', 'Total time'], loc='upper left')

    # Show the plot
    plt.savefig('Imagenet_training_timeBreakdown_PARTIAL.png')


def plot_comp_timeBreakdown(epoch_no, plt_total_comp_output_time, plt_total_comp_loss_time,
                            plt_total_comp_backward_time, plt_total_comp_loss_div_time):

    epochs = len(plt_total_comp_output_time)
    plt_total_time = np.add(np.add(np.add(plt_total_comp_output_time, plt_total_comp_loss_time), plt_total_comp_backward_time), plt_total_comp_loss_div_time)
    barWidth = 0.85

    # Create the stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(epochs), plt_total_comp_output_time, color='#5DA5DA', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_comp_loss_time, bottom=plt_total_comp_output_time, color='#FAA43A', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_comp_backward_time, bottom=np.add(plt_total_comp_output_time, plt_total_comp_loss_time), color='#60BD68', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_comp_loss_div_time, bottom=np.add(np.add(plt_total_comp_output_time, plt_total_comp_loss_time), plt_total_comp_backward_time), color='#F17CB0', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_time, color='grey', alpha=0.2, edgecolor='white', width=barWidth)

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.xticks(range(epochs))
    plt.ylabel('Time (seconds)')
    plt.title('Imagenet Computation Time breakdown per epoch: PARTIAL')
    plt.legend(['Computation output time', 'Computation loss time', 'Computation backward time', 'Computation loss_div time', 'Total time'], loc='upper right')


    # Show the plot
    plt.savefig('Imagenet_Computation_timeBreakdown_PARTIAL.png')


def train(epoch, mini_batch_limit, nc):

    loss_onIndex_onEpoch = dict()

    nc._set_current_epoch(epoch) # IS
    
    rank = hvd.rank()
    world_size = hvd.size()

    acc_res = Result("Accuracy")
    loss_res = Result("Loss")

    model.train()
    _batch_size = configs["MODEL"]["batch_size"] 
    _validation_iteration_number = 7

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")   


    #Inter process communication initialization
    nc.scheduling(epoch)
    
    data_loading_times = list()
    data_loading_start_time = time.time()

    # Synchronize start time across all processes
    loading_start_time = time.time()
    #loading_start_time_acp = hvd.broadcast(torch.tensor(loading_start_time), root_rank=0, name='start_time').item()
    total_computation_time = 0
    total_reading_time = 0

    total_comp_output_time = 0
    total_comp_loss_time = 0
    total_comp_backward_time = 0
    total_comp_loss_div_time = 0
    
    for batch_idx, (data, target, path, index_list_tensor) in enumerate(_train_loader):

        if batch_idx < mini_batch_limit-1:
            # data reading time
            loading_end_time = time.time()

            # index tensor to list
            index_list = index_list_tensor.tolist()

            if _is_cuda:
                data, target = data.cuda(), target.cuda()

            print("Rank#{0}, Epoch#{1}, Mini-batch#{2} Time to read mini-batch: {3} seconds".format(rank, epoch, batch_idx, (loading_end_time - loading_start_time)))

            time_allreduce = hvd.allreduce(torch.tensor(loading_end_time - loading_start_time), average=True)
            total_reading_time += time_allreduce.item()

            #if hvd.rank() == 0:
                #print("Average for all process Time to read mini-batch: {:.2f} seconds".format(time_allreduce.item()))

            #set zero to optimizer
            optimizer.zero_grad()

            #Prepare data for each process by spliting
            train_data_splits = torch.split(data, _batch_size) #create split chunk of size batch size. Then by rank each rank can process a part of chunk of batch_size
            train_target_splits = torch.split(target, _batch_size)

            if len(train_data_splits) > 0:
                cnt = 0
                for i in range(len(train_data_splits)):
                
                    process_train_data = train_data_splits[i] #data to train for ranks
                    process_train_target = train_target_splits[i] #label of those data

                    #tackle last mini batch
                    if len(process_train_data) != _batch_size:
                        continue #currently skipping last batch. Later tackle using proper method
            
                    #check length of data and target are same
                    assert len(process_train_data) == len(process_train_target) , "Error in splitting of data and target "

                    computation_start_time = time.time()
                    computation_output_start_time = time.time()
                    output = model(process_train_data)
                    computation_output_end_time = time.time()
                    
                    comp_output_time_allreduce = hvd.allreduce(torch.tensor(computation_output_end_time - computation_output_start_time), average=True)
                    total_comp_output_time += comp_output_time_allreduce.item()

                    computation_loss_start_time = time.time()
                    loss = loss_fn(output, process_train_target)
                    computation_loss_end_time = time.time()
                    comp_loss_time_allreduce = hvd.allreduce(torch.tensor(computation_loss_end_time - computation_loss_start_time), average=True)
                    total_comp_loss_time += comp_loss_time_allreduce.item()

                    loss_values = []  # List to store the loss values of each sample

                    for i in range(len(output)):
                        individual_output = output[i]
                        individual_target = process_train_target[i]
                        individual_loss = loss_fn(individual_output, individual_target)
                        loss_values.append(individual_loss.item())

                    # Obtain loss for each sample in the mini-batch by index
                    for i, loss_value in enumerate(loss_values):
                        sample_index = index_list[i]
                        sample_loss = loss_values[i]  # Loss of the i-th sample in the mini-batch
                        #print("index:{0}, loss:{1}".format(sample_index, sample_loss))
                        loss_onIndex_onEpoch[sample_index] = sample_loss

                    computation_backward_start_time = time.time()
                    loss.backward()
                    computation_backward_end_time = time.time()
                    comp_backward_allreduce = hvd.allreduce(torch.tensor(computation_backward_end_time - computation_backward_start_time), average=True)
                    total_comp_backward_time += comp_backward_allreduce.item()

                    computation_end_time = time.time()

                    computation_time_allreduce = hvd.allreduce(torch.tensor(computation_end_time - computation_start_time), average=True)
                    total_computation_time += computation_time_allreduce.item()

                    acc = custom_accuracy(output, process_train_target)
      
                    acc_res.update(acc)
                    loss_res.update(loss.item())

        else:
            break
        print(f"Epoch#{epoch}: Rank#{rank} accuracy#{acc} Percent and loss#{loss.item()}")
        print(f"Average Rank#{rank} accuracy#{acc_res.avg()} Percent and loss#{loss_res.avg()}")
        sys.stdout.flush()

        # update model parameters
        hvd.barrier()
        optimizer.step()
        torch.cuda.synchronize()

        plt_train_acc.append(acc)
        plt_train_loss.append(loss.item())
        
        loading_start_time = time.time()

    if rank ==0:
    #    print("Rank#{0}, Epoch#{1}, total average computation time: {2} seconds".format(rank, epoch, total_computation_time))
        print("---- shuffling starts----")

    nc._set_current_unsorted_batchLoss(loss_onIndex_onEpoch) #IS

    shuffling_start_time = time.time()
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

    shuffling_end_time = time.time()
    shuffling_time_allreduce = hvd.allreduce(torch.tensor(shuffling_end_time - shuffling_start_time), average=True)

    if rank==0:
        print("Rank#{0}, Epoch#{1}, Shuffling time: {2} seconds".format(rank, epoch, (shuffling_time_allreduce.item())))
        plt_shuffling_time.append(shuffling_time_allreduce.item())
        plt_comp_time.append(total_computation_time)
        plt_reading_time.append(total_reading_time)
        #loading_start_time_acp = hvd.broadcast(torch.tensor(loading_start_time), root_rank=0, name='start_time').item()

        plt_total_comp_output_time.append(total_comp_output_time)
        plt_total_comp_loss_time.append(total_comp_loss_time)
        plt_total_comp_backward_time.append(total_comp_backward_time)
        plt_total_comp_loss_div_time.append(0)

def validation(epoch):
    #if epoch % 5 = 0:
    model.eval()
    # Compute validation loss and accuracy
    val_loss_met = Result('val_loss')
    val_acc_met = Result('val_accuracy')

    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for (val_data, val_target, path, index_tensor) in _val_loader:
            val_data, val_target = val_data.cuda(), val_target.cuda()
            val_output = model(val_data)
            val_acc = custom_accuracy(val_output, val_target)
            val_loss = loss_fn(val_output, val_target)

            val_acc_met.update(val_acc)
            val_loss_met.update(val_loss.item())

            if rank==0:
                print("````````````````````````````````````````````````")
                print(f"Epoch# {epoch}: Validatio acc#{val_acc} Percent and val loss#{val_loss.item()}")
                print(f"Average Validation Loss: {val_loss_met.avg()}, Validation Accuracy: {val_acc_met.avg()} percent")
                print("`````````````````````````````````````````````````````````````````````")
                sys.stdout.flush()

                plt_val_acc.append(val_acc)
                plt_val_loss.append(val_loss.item())

        model.train()


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

    IMGNET_DIR = configs["ROOT_DATADIR"]["imgnet_dir"]

    
    train_folder = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/train")
    wnids_file = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/wnids.txt")
    words_file = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/words.txt")

    train_dataset = ImageNetDataset(train_folder, wnids_file, words_file, transform=None)
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

    _val_folder = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/val")
    _val_dataset = ImageNetDataset(_val_folder, wnids_file, words_file, transform=None) 

    #custom_sampler = CustomSampler(_val_dataset)
    _val_sampler = dsampler(
            _val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle = True)
    _val_loader = torch.utils.data.DataLoader(
            _val_dataset, batch_size= _batch_size,
            sampler= _val_sampler)

    print("training directory of rank {0} is {1}. training set len {2} and val set len{3}".format(rank, train_folder , len(train_dataset), len(_val_dataset)))

    wd = 0.0001
    use_adasum = 0
    MPI.COMM_WORLD.Barrier()
    model = models.resnet50(pretrained=True)
    num_classes = 1000
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    base_lr = 0.1
    momentum = 0.9
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
    fraction = 0.1
    seed = 41
    
    epoch_no = 10
    total_duration = 0

    nc = ImageNetNodeCommunication(train_dataset, batch_size, fraction, seed, min_train_dataset_len, epoch_no)

    plt_train_acc = list()
    plt_train_loss = list()
    plt_train_time = list()

    plt_val_acc = list()
    plt_val_loss = list()

    plt_comp_time = list()
    plt_reading_time = list()
    plt_shuffling_time = list()

    plt_total_comp_output_time = list()
    plt_total_comp_loss_time = list()
    plt_total_comp_backward_time = list()
    plt_total_comp_loss_div_time = list()

    sample_losses = dict()
    for epoch in range(epoch_no):
        print("------------------- Epoch {0}--------------------\n".format(epoch))
        start_time = time.time()

        train(epoch, mini_batch_limit, nc)

        end_time = time.time()
        duration = end_time - start_time
        total_duration += duration

        hvd.barrier()

        validation(epoch)
        if rank==0:
            print("Iteration {0} took {1:.2f} seconds".format(epoch, duration))
            plt_train_time.append(duration)

        avg_duration = total_duration / epoch_no
        if rank==0:
            print("Average iteration duration: {0:.2f} seconds".format(avg_duration))
        sys.stdout.flush()
 
    # Draw plot
    if rank ==0:
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        fig.suptitle("ACC/LOSS")

        # Plot the training loss and accuracy
        ax[0].plot(np.arange(len(plt_train_loss)), plt_train_loss, label='Training Loss')
        ax[0].plot(np.arange(len(plt_train_acc)), plt_train_acc, label='Training Accuracy')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss / Accuracy')
        ax[0].legend()

        # Plot the validation loss and accuracy
        ax[1].plot(np.arange(len(plt_val_loss)), plt_val_loss, label='Validation Loss')
        ax[1].plot(np.arange(len(plt_val_acc)), plt_val_acc, label='Validation Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss / Accuracy')
        ax[1].legend()

        # Plot the training time
        ax[2].plot(np.arange(len(plt_train_time)), plt_train_time, label='Training Time')
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('Time')
        ax[2].legend()

        plt.savefig('Imagenet_PARTIAL_lossAcc.png')

    if rank == 0:
        plot_timeBreakdown(epoch_no, plt_comp_time, plt_reading_time, plt_shuffling_time)

    if rank == 0:
        plot_comp_timeBreakdown(epoch_no, plt_total_comp_output_time, plt_total_comp_loss_time, plt_total_comp_backward_time, plt_total_comp_loss_div_time)
