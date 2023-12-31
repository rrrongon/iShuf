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
from io import BytesIO
import torch.nn as nn

#File Import
from imagenet_customDatasetInterface import ImageNetDataset
from imagenet_nodeComm import ImageNetNodeCommunication
from imagenetValidation import ImageNetValidationDataset

MINI = "MINI"
_21K = "_21K"
RAND = "RAND"
ISHUF = "iSHUF"

#DATASET = MINI # /21K

'''
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
'''

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
    #accuracy = correct / total

    #print("column max#{0}\n correct#{1}\n accuracy#{2}".format(pred_column_max, correct, accuracy))
    return correct, total

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target):
        # Calculate custom cross-entropy loss for each sample in the batch
        individual_losses = -torch.sum(target * torch.log(F.softmax(input, dim=1)), dim=1)

        if self.weight is not None:
            individual_losses = individual_losses * self.weight

        # Compute the total batch loss
        if self.size_average:
            batch_loss = individual_losses.mean()
        elif self.reduce:
            batch_loss = individual_losses.sum()
        else:
            batch_loss = individual_losses.mean()

        return batch_loss, individual_losses.tolist()


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
    plt.legend(['Reading time', 'Shuffling time', 'Computation time', 'Total time'], loc='upper right')

    # Show the plot
    plt.savefig('Imagenet_training_timeBreakdown_PARTIAL.png')


def plot_comp_timeBreakdown(epoch_no, plt_total_comp_output_time, plt_total_comp_loss_time,
                            plt_total_comp_backward_time, plt_total_comp_loss_div_time, plt_total_isampleComputation_time):

    epochs = len(plt_total_comp_output_time)
    plt_total_time = np.add(np.add(np.add(np.add(plt_total_comp_output_time, plt_total_comp_loss_time), plt_total_comp_backward_time), plt_total_comp_loss_div_time), plt_total_isampleComputation_time)

    barWidth = 0.85

    # Create the stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(epochs), plt_total_comp_output_time, color='#5DA5DA', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_comp_loss_time, bottom=plt_total_comp_output_time, color='#FAA43A', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_comp_backward_time, bottom=np.add(plt_total_comp_output_time, plt_total_comp_loss_time), color='#60BD68', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_comp_loss_div_time, bottom=np.add(np.add(plt_total_comp_output_time, plt_total_comp_loss_time), plt_total_comp_backward_time), color='#F17CB0', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_isampleComputation_time, bottom=np.add(np.add(np.add(plt_total_comp_output_time, plt_total_comp_loss_time), plt_total_comp_backward_time), plt_total_comp_loss_div_time), color='#D7D7D7', edgecolor='white', width=barWidth)  # New bar for plt_total_isampleComputation_time
    plt.bar(range(epochs), plt_total_time, color='grey', alpha=0.2, edgecolor='white', width=barWidth)

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.xticks(range(epochs))
    plt.ylabel('Time (seconds)')
    plt.title('Imagenet Computation Time breakdown per epoch: Isample Partial')

    plt.legend(['Computation output time', 'Computation loss time', 'Computation backward time', 'Computation loss_div time', 'iSample Computation time', 'Total time'], loc='upper right')

    # Show the plot
    plt.savefig('Imagenet_Computation_timeBreakdown_PARTIAL.png')


def train(epoch, mini_batch_limit, nc, _train_sampler, EXP_TYPE):

    loss_onIndex_onEpoch = dict()

    '''
    @ Just set the current epoch to important Sample handler for calculation
    '''
    nc._set_current_epoch(epoch) # this is for important sampling

    _train_sampler.set_epoch(epoch) # For each epoch generate shuffled dataset indices

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

    #loading_start_time_acp = hvd.broadcast(torch.tensor(loading_start_time), root_rank=0, name='start_time').item()
    total_computation_time = 0
    total_reading_time = 0

    total_comp_output_time = 0
    total_comp_loss_time = 0
    total_comp_backward_time = 0
    total_comp_loss_div_time = 0
    isample_comp_time = 0

    total_correct=0
    total_sample = 0

    data_loading_times = list()
    data_loading_start_time = time.time()

    # Synchronize start time across all processes
    loading_start_time = time.time()
    loading_end_time = 0

    #for batch_idx, (data, target, path, index_list_tensor) in enumerate(_train_loader):
    for batch_idx, (data, target, path, index_str_list) in enumerate(_train_loader):
        if batch_idx < mini_batch_limit-1:
            # data reading time
            loading_end_time = time.time()

            if rank ==0:
                if batch_idx % 300==0:
                    print("Epoch#{0}: Mini batch {1}".format(epoch, batch_idx))

            if _is_cuda:
                data, target = data.cuda(), target.cuda()
            #print("Epoch#{0}: Mini batch {1}".format(epoch, batch_idx))
            #set zero to optimizer
            optimizer.zero_grad()

            #Prepare data for each process by spliting
            train_data_splits = torch.split(data, _batch_size) #create split chunk of size batch size. Then by rank each rank can process a part of chunk of batch_size
            train_target_splits = torch.split(target, _batch_size)

            # index tensor to list
            #index_list = index_list_tensor.tolist()

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

                    '''
                    Measure average output calculation time among processes
                    '''
                    computation_output_start_time = time.time()

                    output = model(process_train_data)

                    computation_output_end_time = time.time()
                    comp_output_time_allreduce = hvd.allreduce(torch.tensor(computation_output_end_time - computation_output_start_time), average=True)
                    total_comp_output_time += comp_output_time_allreduce.item()


                    '''
                    Measure loss computation time
                    '''
                    computation_loss_start_time = time.time()
                    #loss = loss_fn(output, process_train_target)
                    loss, loss_values = custom_loss(output, process_train_target)
                    computation_loss_end_time = time.time()

                    #torch.cuda.synchronize()
                    #hvd.allreduce(torch.tensor(0), name="barrier")

                    comp_loss_time_allreduce = hvd.allreduce(torch.tensor(computation_loss_end_time - computation_loss_start_time), average=True)
                    total_comp_loss_time += comp_loss_time_allreduce.item()

            
                    '''
                    Measure backward computation time
                    '''
                    computation_backward_start_time = time.time()
                    loss.backward()
                    computation_backward_end_time = time.time()
                    comp_backward_allreduce = hvd.allreduce(torch.tensor(computation_backward_end_time - computation_backward_start_time), average=True)
                    total_comp_backward_time += comp_backward_allreduce.item()

                    #loss_values = []  # List to store the loss values of each sample
                    i_list = list()

                    isample_computation_start_time = time.time()

                    if EXP_TYPE == ISHUF:
                        #for i in range(len(output)):
                        #    individual_output = output[i]
                        #    individual_target = process_train_target[i]
                        #    individual_loss = loss_fn(individual_output, individual_target)
                        #    loss_values.append(individual_loss.item())

                        # Obtain loss for each sample in the mini-batch by index
                        for i, loss_value in enumerate(loss_values):
                            #sample_index = index_list[i]
                            sample_index = index_str_list[i]
                            sample_index = int(sample_index)
                            sample_loss = loss_value #loss_values[i]  # Loss of the i-th sample in the mini-batch
                            #print("index:{0}, loss:{1}".format(sample_index, sample_loss))
                            loss_onIndex_onEpoch[sample_index] = sample_loss
                            i_list.append(sample_index)

                    isample_computation_end_time = time.time()
                    isample_comp_allreduce = hvd.allreduce(torch.tensor(isample_computation_end_time - isample_computation_start_time), average=True)
                    isample_comp_time += isample_comp_allreduce.item()

                    correct, mini_batch_total = custom_accuracy(output, process_train_target)
                    total_correct += correct
                    total_sample += mini_batch_total

                   # acc_res.update(acc)
                   # loss_res.update(loss.item())

        else:
            break

        # update model parameters
        hvd.barrier()
        optimizer.step()
        torch.cuda.synchronize()
        hvd.allreduce(torch.tensor(0), name="barrier")

        '''
        @Calculate reading time for current mini-batchs and keep adding to calculate total reading time for an epoch
        '''
        time_allreduce = hvd.allreduce(torch.tensor(loading_end_time - loading_start_time), average=True)
        total_reading_time += time_allreduce.item()

        loading_start_time = time.time()

    if rank ==0:
    #    print("Rank#{0}, Epoch#{1}, total average computation time: {2} seconds".format(rank, epoch, total_computation_time))
        print("---- shuffling starts----")

    train_dataset.set_tranforms(None)

    #Get reading time from the dataset internal
    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")
    true_img_reading_time = train_dataset.image_reading_times
    plt_image_reading_times.append(train_dataset.image_reading_times)
    train_dataset.image_reading_times = 0
    
    #hvd.allreduce(torch.tensor(0), name="barrier")
    true_img_reading_time_allreduce = hvd.allreduce(torch.tensor(true_img_reading_time), average=True)
    true_img_reading_time_allr = true_img_reading_time_allreduce.item() 
    plt_true_img_reading_time_allreduce.append(true_img_reading_time_allr)

    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")
    true_image_readTrans_time = train_dataset.image_readTrans_times
    plt_image_readtrans_times.append(train_dataset.image_readTrans_times)
    train_dataset.image_readTrans_times = 0
    
    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")
    true_image_readTrans_time_allreduce = hvd.allreduce(torch.tensor(true_image_readTrans_time), average=True)
    true_image_readTrans_time_allr = true_image_readTrans_time_allreduce.item()
    plt_true_image_readTrans_time_allreduce.append(true_image_readTrans_time_allr)

    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")
    epoch_total_fileSize = train_dataset.total_fileSize
    train_dataset.total_fileSize = 0
    avg_epoch_total_fileSize = hvd.allreduce(torch.tensor(epoch_total_fileSize), average=True)
    avg_epoch_total_fileSize = avg_epoch_total_fileSize.item()
    plt_avg_epoch_total_fileSize.append(avg_epoch_total_fileSize)

    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")
    epoch_fileAccess_count = train_dataset.count_fileAccess
    train_dataset.count_fileAccess = 0
    avg_epoch_fileAccess_count = hvd.allreduce(torch.tensor(epoch_fileAccess_count), average=True)
    avg_epoch_fileAccess_count = avg_epoch_fileAccess_count.item()
    plt_avg_epoch_fileAccess_count.append(avg_epoch_fileAccess_count)


    torch.cuda.synchronize()
    hvd.allreduce(torch.tensor(0), name="barrier")

    shuffling_start_time = time.time()
    nc._set_current_unsorted_batchLoss(loss_onIndex_onEpoch) #IS
    nc.scheduling(epoch)
    nc._communicate()
    print("Rank#{0} is done in Communicating".format(rank))
    sys.stdout.flush()
    torch.cuda.synchronize()
    hvd.allreduce(torch.tensor(0), name="barrier")

    nc.sync_recv()
    #print("Rank#{0} is done in sync recv".format(rank))
    sys.stdout.flush()
    torch.cuda.synchronize()
    hvd.allreduce(torch.tensor(0), name="barrier")

    nc.sync_send()
    #print("Rank#{0} is done in sync send".format(rank))
    sys.stdout.flush()
    torch.cuda.synchronize()
    hvd.allreduce(torch.tensor(0), name="barrier")

    nc.clean_sent_samples()
    #print("Rank#{0} is done in clean sent samples".format(rank))
    sys.stdout.flush()
    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")

    shuffling_end_time = time.time()
    shuffling_time_allreduce = hvd.allreduce(torch.tensor(shuffling_end_time - shuffling_start_time), average=True)

    '''
    @Measure total computation time by adding segmented sum of times
    Measure total training time by adding computation time, reading time, shuffling time
    '''
    computation_time = total_comp_output_time + total_comp_loss_time + total_comp_backward_time + isample_comp_time
    training_time = total_reading_time + computation_time + shuffling_time_allreduce.item()

    #correct = custom_accuracy(output, process_train_target)
    acc = (total_correct/total_sample) * 100
    acc_res.update(acc)
    loss_res.update(loss.item())

    plt_train_acc.append(acc)
    plt_train_loss.append(loss.item())

    print(f"Epoch#{epoch}: Rank#{rank} accuracy#{acc} Percent and loss#{loss.item()}")
    sys.stdout.flush()

    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")

    plt_shuffling_image_reading_times.append(train_dataset.image_reading_times)
    train_dataset.image_reading_times = 0
    plt_shuffling_image_readtrans_times.append(train_dataset.image_readTrans_times)
    train_dataset.image_readTrans_times=0

    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")

    epoch_total_shuf_fileSize = train_dataset.total_fileSize
    train_dataset.total_fileSize = 0
    avg_epoch_total_shuf_fileSize = hvd.allreduce(torch.tensor(epoch_total_shuf_fileSize), average=True)
    avg_epoch_total_shuf_fileSize = avg_epoch_total_shuf_fileSize.item()
    plt_avg_epoch_total_shuf_fileSize.append(avg_epoch_total_shuf_fileSize)


    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")
    epoch_fileAccess_count_shuf = train_dataset.count_fileAccess
    train_dataset.count_fileAccess = 0
    avg_epoch_fileAccess_count_shuf = hvd.allreduce(torch.tensor(epoch_fileAccess_count_shuf), average=True)
    avg_epoch_fileAccess_count_shuf = avg_epoch_fileAccess_count_shuf.item()
    plt_avg_epoch_fileAccess_count_shuf.append(avg_epoch_fileAccess_count_shuf)


    if rank==0:
        print(f"Average Rank#{rank} accuracy#{acc_res.avg()} Percent and loss#{loss_res.avg()}")
        sys.stdout.flush()

        print("Rank#{0}, Epoch#{1}, Shuffling time: {2} seconds".format(rank, epoch, (shuffling_time_allreduce.item())))
        plt_shuffling_time.append(shuffling_time_allreduce.item())
        plt_comp_time.append(computation_time)
        plt_reading_time.append(total_reading_time)
        #loading_start_time_acp = hvd.broadcast(torch.tensor(loading_start_time), root_rank=0, name='start_time').item()

        plt_total_comp_output_time.append(total_comp_output_time)
        plt_total_comp_loss_time.append(total_comp_loss_time)
        plt_total_comp_backward_time.append(total_comp_backward_time)
        plt_total_comp_loss_div_time.append(0)
        plt_total_isampleComputation_time.append(isample_comp_time)

        plt_total_training_time.append(training_time)
        plt_total_computation_time.append(computation_time)
        plt_train_time.append(training_time)

    #torch.cuda.synchronize()
    #hvd.allreduce(torch.tensor(0), name="barrier")

def validation(epoch):
    #if epoch % 5 = 0:
    model.eval()
    # Compute validation loss and accuracy
    val_loss_met = Result('val_loss')
    val_acc_met = Result('val_accuracy')

    val_loss = 0.0
    val_acc = 0.0

    total_correct = 0
    total_sample = 0
    with torch.no_grad():
        for (val_data, val_target) in _val_loader:
            val_data, val_target = val_data.cuda(), val_target.cuda()
            val_output = model(val_data)
            val_loss = loss_fn(val_output, val_target)
            corr, samples = custom_accuracy(val_output, val_target)
            total_correct += corr
            total_sample += samples

        val_acc = (total_correct/total_sample ) * 100
        val_acc_met.update(val_acc)
        val_loss_met.update(val_loss.item())

        print("````````````````````````````````````````````````")
        print(f"Epoch# {epoch}: Validatio acc#{val_acc} Percent and val loss#{val_loss.item()}")
        print(f"Average Validation Loss: {val_loss_met.avg()}, Validation Accuracy: {val_acc_met.avg()} percent")
        print("`````````````````````````````````````````````````````````````````````")
        sys.stdout.flush()

        if rank==0:
            print("````````````````````````````````````````````````")
            print(f"Epoch# {epoch}: Validatio acc#{val_acc} Percent and val loss#{val_loss.item()}")
            print(f"Average Validation Loss: {val_loss_met.avg()}, Validation Accuracy: {val_acc_met.avg()} percent")
            print("`````````````````````````````````````````````````````````````````````")
            sys.stdout.flush()

            plt_val_acc.append(val_acc)
            plt_val_loss.append(val_loss.item())

        model.train()

def calculate_average(lst):
    if len(lst) == 0:
        return 0  # Handle the case when the list is empty

    total = sum(lst)
    average = total / len(lst)
    return average


if __name__ == '__main__':

    _is_cuda = torch.cuda.is_available()
    if not _is_cuda:
        print("There is no CUDA available\n")
        exit(1)

    hvd.init()
    rank = hvd.rank()

    torch.cuda.set_device(rank)

    f = open('/home/r.rongon/research/project_shuffle/customdatasampler_rand/config.json')
    configs =json.load(f)
    torch.manual_seed(configs["MODEL"]["seed"])

    DATASET = configs["DATA_TYPE"]
    EXP_TYPE = configs["EXP_TYPE"]

    if DATASET == MINI:
        IMGNET_DIR = configs["ROOT_DATADIR"]["imgnet_dir"]
        val_wnid_file = "/scratch/user/r.rongon/dataset_20231122_103233/imagenet_dataset/imagenet-mini/val_label_mini.txt"
        OUT_FOLDER = configs["mini"]
        PARTITION_DIR = configs["mini"] 
        TARGET_DIR =configs["mini"] 
        CLASS_NUMBER = 1000

    elif DATASET == _21K:
        IMGNET_DIR = configs["ROOT_DATADIR"]["imgnet_21k_dir"]
        val_wnid_file = "/scratch/user/r.rongon/dataset_20231122_103233/imagenet_dataset/imagenet21k/val_label_mini.txt" #"./imagenet_dataset/imagenet21k_resized/ImageNet_val_label.txt"
        OUT_FOLDER = configs["_21K"]  #'./imagenet_dataset/imagenet21k_resized'
        PARTITION_DIR = configs["_21K"]  #'./imagenet_dataset/imagenet21k_resized'
        TARGET_DIR = configs["_21K"] # './imagenet_dataset/imagenet21k_resized'
        CLASS_NUMBER = 21844

    train_folder = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/train")
    wnids_file = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/wnids.txt")
    words_file = os.path.join(IMGNET_DIR,"parition" + str(rank)+"/words.txt")

    #val_wnid_file = os.path.join(IMGNET_DIR,"ImageNet_val_label.txt")
    class_label_file = os.path.join(IMGNET_DIR,"class-label.txt")
    _val_folder = "/scratch/user/r.rongon/dataset_20231122_103233/imagenet_dataset/val_dataset"

    #val_wnid_file = "./imagenet_dataset/imagenet21k_resized/ImageNet_val_label.txt"


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

    is_preTrain = configs["PreTrain"]
    use_adasum = 0
    MPI.COMM_WORLD.Barrier()

    if is_preTrain:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)

    num_classes = CLASS_NUMBER
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    '''
    model = models.alexnet(pretrained=True)

    num_classes = CLASS_NUMBER
    num_ftrs = model.classifier[6].in_features  # Access the last fully connected layer of AlexNet
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    '''
    base_lr = configs["MODEL"]["base_lr"]
    momentum = configs["MODEL"]["moment"]

    scaled_lr = base_lr #* hvd.size()
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

    custom_loss = CustomCrossEntropyLoss()

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
    plt_total_isampleComputation_time = list()

    plt_total_training_time = list()
    plt_total_computation_time = list()

    plt_image_reading_times = list()
    plt_image_readtrans_times = list()

    plt_shuffling_image_reading_times = list()
    plt_shuffling_image_readtrans_times = list()

    plt_true_img_reading_time_allreduce = list()
    plt_true_image_readTrans_time_allreduce = list()

    plt_avg_epoch_total_fileSize = list()
    plt_avg_epoch_total_shuf_fileSize = list()

    plt_avg_epoch_fileAccess_count_shuf = list()
    plt_avg_epoch_fileAccess_count = list()

    sample_losses = dict()
    for epoch in range(epoch_no):
        print("------------------- Epoch {0}--------------------\n".format(epoch))
        #hvd.barrier()
        if epoch !=0 and epoch % 6==0:
            scaled_lr *= 0.1
            print("learning rate changed to {0}".format(scaled_lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = scaled_lr
        #hvd.barrier()
        
        train(epoch, mini_batch_limit, nc, _train_sampler, EXP_TYPE)

        #torch.cuda.synchronize()
        #hvd.allreduce(torch.tensor(0), name="barrier")
        hvd.barrier()

        train_dataset.set_tranforms(transforms.Compose([
            transforms.Resize(256),                          # Resize the image to 256x256 pixels
            transforms.CenterCrop(224),                      # Crop the center 224x224 pixels
            transforms.Grayscale(num_output_channels=3),     # Convert the image to RGB if it's grayscale
            transforms.ToTensor(),                            # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image channels
                                std=[0.229, 0.224, 0.225])
        ]))
        validation(epoch)
        #plt_image_reading_times.append(train_dataset.image_reading_times)
        #train_dataset.image_reading_times = 0
        #plt_image_readtrans_times.append(train_dataset.image_readTrans_times)
        #train_dataset.image_readTrans_times = 0
        hvd.barrier()
        #torch.cuda.synchronize()
        #hvd.allreduce(torch.tensor(0), name="barrier")

    nc.dump_result(rank)
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
        plot_comp_timeBreakdown(epoch_no, plt_total_comp_output_time, plt_total_comp_loss_time, plt_total_comp_backward_time, plt_total_comp_loss_div_time, plt_total_isampleComputation_time)

    if rank ==0:
        '''
        @Calculate average time for each segment
        '''
        avg_shuffling_time = calculate_average(plt_shuffling_time)
        avg_reading_time = calculate_average(plt_reading_time)
        #loading_start_time_acp = hvd.broadcast(torch.tensor(loading_start_time), root_rank=0, name='start_time').item()

        avg_output_time = calculate_average(plt_total_comp_output_time)
        avg_loss_time = calculate_average(plt_total_comp_loss_time)
        avg_back_time = calculate_average(plt_total_comp_backward_time)
        avg_training_time = calculate_average(plt_total_training_time)
        avg_comp_time= calculate_average(plt_total_computation_time)
        avg_isample_com_time = calculate_average(plt_total_isampleComputation_time)

        avg_img_reading_time = calculate_average(plt_image_reading_times)
        avg_readTrans_time = calculate_average(plt_image_readtrans_times)
        avg_shuffling_img_reading_time = calculate_average(plt_shuffling_image_reading_times)
        avg_shuffling_readTrans_time = calculate_average(plt_shuffling_image_readtrans_times)

        avg_true_img_reading_time_allreduce = calculate_average(plt_true_img_reading_time_allreduce)
        avg_true_image_readTrans_time_allreduce = calculate_average(plt_true_image_readTrans_time_allreduce)

        avg_plt_avg_epoch_total_fileSize = calculate_average(plt_avg_epoch_total_fileSize)
        avg_plt_avg_epoch_total_shuf_fileSize = calculate_average(plt_avg_epoch_total_shuf_fileSize)

        avg_plt_avg_epoch_fileAccess_count_shuf = calculate_average(plt_avg_epoch_fileAccess_count_shuf)
        avg_plt_avg_epoch_fileAccess_count = calculate_average(plt_avg_epoch_fileAccess_count)

        print("----------------------\n")
        print("Average training time: {0}\n".format(avg_training_time))
        print("Average computation time: {0}\n".format(avg_comp_time))
        print("Average reading time: {0}\n".format(avg_reading_time))
        print("Average shuffling time: {0}\n".format(avg_shuffling_time))
        print("Average output time: {0}\n".format(avg_output_time))
        print("Average loss time: {0}\n".format(avg_loss_time))
        print("Average backprop time: {0}\n".format(avg_back_time))
        print("Average iSample computation time: {0}\n".format(avg_isample_com_time))

        print("Average actual reading time from dataset class: {0}".format(avg_img_reading_time))
        print("Average actual read+Transformation time from dataset class: {0}\n".format(avg_readTrans_time))

        print("Average reading time while shuffling dataset class: {0}".format(avg_shuffling_img_reading_time))
        print("Average reading+ transformation time while shuffling dataset class: {0}\n".format(avg_shuffling_readTrans_time))

        print("Average Average reading time while shuffling dataset class: {0}".format(avg_true_img_reading_time_allreduce))
        print("Average Average reading + transformation time while shuffling dataset class: {0}\n".format(avg_true_image_readTrans_time_allreduce))

        print("Average Average total file size in KB during reading time: {0}".format(avg_plt_avg_epoch_total_fileSize))
        print("Average Average total file size during shuffling in KB: {0}\n".format(avg_plt_avg_epoch_total_shuf_fileSize))

        print("Average Average total file access count during training: {0}".format(avg_plt_avg_epoch_fileAccess_count))
        print("Average Average total file access count during shuffling: {0}\n".format(avg_plt_avg_epoch_fileAccess_count_shuf))
