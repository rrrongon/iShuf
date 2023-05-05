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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#File Import
from InterNodeCommunication import NodeCommunication

'''
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n
'''

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path

#training_data = datasets.FashionMNIST(
#    root="data",
#    train=True,
#    download=True,
#    transform=ToTensor()
#)

#test_data = datasets.FashionMNIST(
#    root="data",
#    train=False,
#    download=True,
#    transform=ToTensor()
#)
'''
training_data = CustomImageDataset(annotations_file='/home/rongon/Downloads/archive/All.csv',
 img_dir='/home/rongon/Downloads/archive/all_images/')

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    print(len(training_data))
    print("\n")
    print(sample_idx)
    img, label = training_data[sample_idx]
    print("type image: ")
    print(type(img))
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.numpy()[0], cmap="gray")
plt.show()'''


#rootdir root_dir = configs["ROOT_DATADIR"]["train_dir"]
#_file = root_dir +_file_name  + "_filepath.csv"

def __get_image_count(img_dir):
    os.listdir(img_dir)

def rank_replace_img_path(rank, old_path):
    # split the path into its components
    path_components = old_path.split("/")
    # replace the 3rd component with "rank_data_2"
    path_components[3] = str(rank)+"_data_2"

    # reconstruct the path
    new_path = os.path.join(*path_components)
    #print("Old path#{0} and new path#{1}".format(old_path, new_path))
    return new_path

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_file_path, transform = None, target_transform=None):
        super().__init__()
        self.img_dir = img_dir
        try:
            self._labels = pd.read_csv(label_file_path)
            #df = self._labels.head(820)
            #self._labels = df
            #df = None
            #random_df = self._labels.sample(n=300, random_state=42)
            #self._labels = random_df
            #random_df = None
        except:
            print("Could not open file {0}".format(label_file_path))
            exit(1)
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self._labels.iloc[idx,0])
        except Exception as e:
            print(f"self._labels.iloc[idx,0]#{self._labels.iloc[idx,0]}\n self.img_dir#{self.img_dir}")
            exit(1)
        image = PIL.Image.open(img_path)
        label = self._labels.iloc[idx, 1]
        new_label = str(label) + "_" + str(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        y_one_hot = torch.zeros(10)
        y_one_hot[label] = 1
        return image, y_one_hot, img_path

    def add_new_item(self, rank, img_tensor, idx, path, label):

        #after receiving an item from another node via MPI need to add to existing dataset object to train
        replaced_path = rank_replace_img_path(rank, path)
        new_item = {'path': replaced_path.encode('utf-8'), 'label': label}
        #self._labels.loc[n] = new_item
        self._labels = self._labels.append(new_item, ignore_index=True)
        #print("Rank# {0} is adding labels and item detail {1}\n".format(rank, new_item))
        #tensor_to_image = transforms.ToPILImage()
        #image = tensor_to_image(img_tensor)

        # Save image to directory
        #image.save(path)
        #print("Rank#{0} Image saved path {1}\n".format(rank, path))

    def add_new_samples(self, rank, recvd_samples):
        new_items = list()
        try:
            for sample in recvd_samples:
                path = sample['path']
                label = torch.argmax(sample['class_name']).item()
                replaced_path = rank_replace_img_path(rank, path) #'./natural_image/Partition_Folder/0_data_2/train/motorbike/motorbike_0687_1.jpg'
                partial_path = replaced_path.split('train/')[1] # 'motorbike/motorbike_0687_1.jpg'
                new_item = {'path': str(partial_path), 'label': label}
                new_items.append(new_item)
            #print("Before label size of rank#{0}: {1}".format(rank, self._labels.size ))
            self._labels = pd.concat([self._labels, pd.DataFrame(new_items)], ignore_index=True)
            #print("After addinf items label size of rank#{0}: {1}".format(rank, self._labels.size ))
            sys.stdout.flush()
        except Exception as e:
            print("Error in adding new samples in rank#{0}".format(rank))
            print("Exception on Rank#{0}".format(str(e)))
            sys.stdout.flush()
            hvd.Abort()

        #Need to save the tensors to image path
        try:
            for sample in recvd_samples:
                tensor_to_image = transforms.ToPILImage()
                image = tensor_to_image(sample['sample'])
                path = str(sample['path'])
                replaced_path = rank_replace_img_path(rank, path)
                image.save(replaced_path)
                #print("Rank#{0} Image saved path {1}\n".format(rank, path))
                sys.stdout.flush()
        except Exception as e:
            print("Error in saving Image in rank#{0} and path#{1}".format(rank, rank_replace_img_path(rank, path)))
            print("Exception on Rank#{0}".format(str(e)))
            sys.stdout.flush()
            hvd.Abort()

    def delete_an_item(self, rank, idx):
        print("Rank#{0} is trying to remove an item with index#{1}".format(rank, idx))

        #remove from label dataframe
        self._labels = self._labels.drop(idx)
        #remove from storage
        img_path = os.path.join(self.img_dir, self._labels.iloc[idx,0])
        print("Rank#{0} deleted the image from storage path#{1}. Please check...".format(rank, path))
        os.remove(img_path)

    def remove_old_samples(self, rank, clean_list):
        full_paths = list()
        #before dropping copy the directory of the images
        selected_rows = self._labels.iloc[clean_list]
        try:
            for index, row in selected_rows.iterrows():
                path = row.iloc[0]
                img_path = os.path.join(self.img_dir, path)
                full_paths.append(img_path)

            #drop from the dataframe
            #print("Before dropping label size of rank#{0}: {1}".format(rank, self._labels.size ))
            df = self._labels.drop(clean_list) 
            self._labels=None
            self._labels = df
            df = None
            #print("After dropping label size of rank#{0}: {1}".format(rank, self._labels.size ))

            #delete images from the local storage
            for file_path in full_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    #print(f"Rank#{rank} Deleted file: {file_path}")
                else:
                    print(f"Rank#{rank} File {file_path} does not exist")
        except Exception as e:
            print(f" Rank#{rank} Exception# {str(e)}")
            exit(1)

def get_accuracy(output, target):
    # get the index of the max log-probability
    print("Output tensor#{0}, output shape#{1} \n target tensor#{2}, target shape#{3}".format(output, output.size(),target,target.size()))
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

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        async_handle = hvd.allreduce_async(val, average = False)
        self.n += 1
        return async_handle

    @property
    def avg(self):
        return self.sum / self.n

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

def plot_timeBreakdown(epochs, plt_comp_time, plt_reading_time):

    plt_total_time = np.add(plt_comp_time, plt_reading_time)
    barWidth = 0.85

    # Create the stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(epochs), plt_reading_time, color='#5DA5DA', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_comp_time, bottom=plt_reading_time, color='#60BD68', edgecolor='white', width=barWidth)
    plt.bar(range(epochs), plt_total_time, color='grey', alpha=0.2, edgecolor='white', width=barWidth)

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.xticks(range(epochs))
    plt.ylabel('Time (seconds)')
    plt.title('Time breakdown per epoch')
    plt.legend(['Reading time', 'Computation time', 'Total time'], loc='upper left')


    # Show the plot
    plt.savefig('basic_time_breakdown.png')


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
    plt.title('Time breakdown per epoch')
    plt.legend(['Computation output time', 'Computation loss time', 'Computation backward time', 'Computation loss_div time', 'Total time'], loc='upper right')


    # Show the plot
    plt.savefig('Computation_breakdown_basic.png')


import torch.nn as nn

def train(epoch):

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
    print(f"Rank#{rank} available device#{device}")


    #Inter process communication initialization
    #nc.scheduling(epoch)
    
    loading_start_time = time.time()
    #loading_start_time_acp = hvd.broadcast(torch.tensor(loading_start_time), root_rank=0, name='start_time').item()
    total_computation_time = 0
    total_reading_time = 0

    total_comp_output_time = 0
    total_comp_loss_time = 0
    total_comp_backward_time = 0
    total_comp_loss_div_time = 0

    for batch_idx, (data, target, path) in enumerate(_train_loader):
        # data reading time
        loading_end_time = time.time()
        total_reading_time += (loading_end_time - loading_start_time)

        if _is_cuda:
            data, target = data.cuda(), target.cuda()
        
        #set zero to optimizer
        optimizer.zero_grad()

        #Prepare data for each process by spliting
        train_data_splits = torch.split(data, _batch_size) #create split chunk of size batch size. Then by rank each rank can process a part of chunk of batch_size
        train_target_splits = torch.split(target, _batch_size)

        if len(train_data_splits) > 0:
            cnt = 0
            for i in range(len(train_data_splits)):
                
                if i % hvd.size() ==  rank:
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

                    total_comp_output_time += (computation_output_end_time - computation_output_start_time)

                    computation_loss_start_time = time.time()
                    loss = loss_fn(output, process_train_target)
                    computation_loss_end_time = time.time()

                    total_comp_loss_time += (computation_loss_end_time - computation_loss_start_time)

                    # compute gradients
                    computation_backward_start_time = time.time()
                    loss.backward()
                    computation_backward_end_time = time.time()

                    total_comp_backward_time += (computation_backward_end_time - computation_backward_start_time)

                    computation_loss_div_start_time = time.time()
                    loss.div_( (hvd.size()*math.ceil(float(len(data)))) / (_batch_size/hvd.size()))
                    computation_loss_div_end_time = time.time()

                    total_comp_loss_div_time += (computation_loss_div_end_time - computation_loss_div_start_time)

                    computation_end_time = time.time()
                    total_computation_time += (computation_end_time - computation_start_time)

                    acc = custom_accuracy(output, process_train_target)

        
                    acc_res.update(acc)
                    loss_res.update(loss.item())

            #    if (batch_idx + 1) % _validation_iteration_number == 0:
            #        model.eval()
                    # Compute validation loss and accuracy
            #        val_loss_met = Result('val_loss')
            #        val_acc_met = Result('val_accuracy')
                    
            #        val_loss = 0.0
            #        val_acc = 0.0

            #        with torch.no_grad():
            #            for (val_data, val_target, path) in _val_loader:
            #                val_data, val_target = val_data.cuda(), val_target.cuda()
            #                val_output = model(val_data)
            #                val_acc = custom_accuracy(val_output, val_target)
            #                val_loss = loss_fn(val_output, val_target)
                            
            #                val_acc_met.update(val_acc)
            #                val_loss_met.update(val_loss.item())
                    
            #            if rank==0:
            #                print(f"Validation Loss: {val_loss_met.avg()}, Validation Accuracy: {val_acc_met.avg()}")
            
            #        model.train()

        print(f"Epoch#{epoch}: Rank#{rank} accuracy#{acc} and loss#{loss.item()}")
        print(f"Average Rank#{rank} accuracy#{acc_res.avg()} and loss#{loss_res.avg()}")

        # update model parameters
        hvd.barrier()
        optimizer.step()

        torch.cuda.synchronize()

        #check model param accross different ranks
        #for param_name, param in model.named_parameters():
        #    print(f"Rank#{rank} model param name#{param_name} and value#{param}\n")

        #Internode sync
        #nc._communicate(epoch)
        #torch.cuda.synchronize()

        #nc.sync_recv()
        #torch.cuda.synchronize()

        #nc.sync_send()
        #torch.cuda.synchronize()

        #nc.clean_sent_samples()
        #torch.cuda.synchronize()

        plt_train_acc.append(acc)
        plt_train_loss.append(loss.item())

        loading_start_time = time.time()

    if rank==0:
        plt_comp_time.append(total_computation_time)
        plt_reading_time.append(total_reading_time)

        plt_total_comp_output_time.append(total_comp_output_time)
        plt_total_comp_loss_time.append(total_comp_loss_time)
        plt_total_comp_backward_time.append(total_comp_backward_time)
        plt_total_comp_loss_div_time.append(total_comp_loss_div_time)

        '''
            #accuracy_iter = accuracy(output, target_batch)
            _, predicted = torch.max(output, 1)
            _, target = torch.max(target_batch, 1)
            try:
                correct = (predicted == target).sum().item()
            except Exception as e:
                print("Exception{0}".format(e))
                print("predicted{0}\n target_batch: {1}".format(predicted, target_batch))
            accuracy = correct / len(target_batch)
            print("Accuracy: {:.2f}%".format(accuracy*100))

            train_accuracy.update(accuracy)
            train_loss.update(loss)

            loss.div_(math.ceil(float(len(data)) / _batch_size))
            loss.backward()
        '''
def validation(epoch):
    #if epoch % 5 = 0:
    model.eval()
    # Compute validation loss and accuracy
    val_loss_met = Result('val_loss')
    val_acc_met = Result('val_accuracy')

    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for (val_data, val_target, path) in _val_loader:
            val_data, val_target = val_data.cuda(), val_target.cuda()
            val_output = model(val_data)
            val_acc = custom_accuracy(val_output, val_target)
            val_loss = loss_fn(val_output, val_target)

            val_acc_met.update(val_acc)
            val_loss_met.update(val_loss.item())

            if rank==0:
                print("````````````````````````````````````````````````")
                print(f"Epoch# {epoch}: Validatio acc#{val_acc} and val loss#{val_loss.item()}")
                print(f"Average Validation Loss: {val_loss_met.avg()}, Validation Accuracy: {val_acc_met.avg()}")
                print("`````````````````````````````````````````````````````````````````````")
                sys.stdout.flush()

                plt_val_acc.append(val_acc)
                plt_val_loss.append(val_loss.item())

        model.train()


if __name__ == '__main__':
    
    hvd.init()
    rank = hvd.rank()
    f = open('config.json')
    configs =json.load(f)
    torch.manual_seed(configs["MODEL"]["seed"])

    root_dir = configs["ROOT_DATADIR"]["train_dir"]
    '''
    -partial
    root_dir = configs["ROOT_DATADIR"]["partiion_dir"]
    root_dir = root_dir + str(rank) + "_data_2/"
    '''

    _train_dir = root_dir + "train/"
    _label_file_path = os.path.join(root_dir, "train_filepath.csv")
    _train_dataset = CustomDataset(img_dir=_train_dir, label_file_path=_label_file_path)

    _no_of_intraNode_workers = hvd.size() #configs["MODEL"]["no_of_batches"]
    _batch_size = configs["MODEL"]["batch_size"]

    _intraNode_batch_size = _no_of_intraNode_workers * _batch_size

    _is_cuda = torch.cuda.is_available()
    if not _is_cuda:
        print("There is no CUDA available\n")
        exit(1)

    #custom_sampler = CustomSampler(_train_dataset)
    _train_sampler = dsampler(
            _train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
    _train_loader = torch.utils.data.DataLoader(
            _train_dataset, batch_size=_intraNode_batch_size,
            sampler= _train_sampler)

    '''
    print("train loader:\n")
    it = iter(_train_loader)
    print(next(it))
    '''
    _val_dir = root_dir + "val/"
    _label_file_path = os.path.join(root_dir, "val_filepath.csv")
    _val_dataset = CustomDataset(img_dir=_val_dir, label_file_path=_label_file_path)

    #custom_sampler = CustomSampler(_val_dataset)
    _val_sampler = dsampler(
            _val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle = True)
    _val_loader = torch.utils.data.DataLoader(
            _val_dataset, batch_size= _intraNode_batch_size,
            sampler= _val_sampler)
    '''
    print("test loader:\n")
    it = iter(_val_loader)
    print(next(it))
    '''
    print("training directory of rank {0} is {1}. training set len {2} and val set len{3}".format(rank,root_dir,len(_train_dataset), len(_val_dataset)))

    wd = 0.0005
    use_adasum = 0
    MPI.COMM_WORLD.Barrier()
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    base_lr = 0.0125
    momentum = 0.2
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

    '''
    #Inter node communication initialization
    batch_size = configs["MODEL"]["batch_size"]
    fraction = 0.2
    seed = 41
    nc = NodeCommunication(_train_dataset, batch_size, fraction, seed)
    '''

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

    epoch_no = 5
    total_duration = 0

    for epoch in range(epoch_no):
        print("------------------- Epoch {0}--------------------\n".format(epoch))
        start_time = time.time()

        train(epoch)

        end_time = time.time()
        duration = end_time - start_time
        total_duration += duration

        hvd.barrier()
        if rank==0:
            print("Iteration {0} took {1:.2f} seconds".format(epoch, duration))
            plt_train_time.append(duration)

        avg_duration = total_duration / epoch_no
        if rank==0:
            print("Average iteration duration: {0:.2f} seconds".format(avg_duration))
        
    # Draw plot
    if rank ==0:
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        fig.suptitle("Process Results")

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

        plt.savefig('Basic_Shuffling.png')

    if rank == 0:
        plot_timeBreakdown(epoch_no, plt_comp_time, plt_reading_time)

    if rank == 0:
        plot_comp_timeBreakdown(epoch_no, plt_total_comp_output_time, plt_total_comp_loss_time, plt_total_comp_backward_time, plt_total_comp_loss_div_time)
