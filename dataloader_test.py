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
        return image, label

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

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_file_path, transform = None, target_transform=None):
        super().__init__()
        self.img_dir = img_dir
        try:
            self._labels = pd.read_csv(label_file_path)
        except:
            print("Could not open file {0}".format(label_file_path))
            exit(1)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self._labels.iloc[idx,0])
        image = PIL.Image.open(img_path)
        label = self._labels.iloc[idx, 1]
        new_label = str(label) + "_" + str(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        y_one_hot = torch.zeros(10)
        y_one_hot[label] = 1
        return image, y_one_hot

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

def train(epoch):
    torch.cuda.synchronize()
    model.train()

    #train_dataset.next_epoch() ## Update the dataset to use the newly received samples
    #train_sampler.set_epoch(epoch) ## Set the epoch in sampler and #Create a new indices list
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    rank = hvd.rank()
    _batch_size = configs["MODEL"]["batch_size"]

    torch.cuda.synchronize()
    for batch_idx, (data, target) in enumerate(_train_loader):
        print("batch no: {0}, length of batch: {1}, target len: {2}, each target len{3}".format(batch_idx, len(data), len(target), len(target[0])))
        if _is_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        for i in range(0, len(data), _batch_size)
            data_batch = data[i:i + _batch_size]
            target_batch = target[i:i + _batch_size]
            output = model(data_batch)
            loss = F.cross_entropy(output, target_batch)

            torch.cuda.synchronize()
            accuracy_iter = accuracy(output, target_batch)
            train_accuracy.update(accuracy_iter)
            train_loss.update(loss)

            loss.div_(math.ceil(float(len(data)) / _batch_size)
            loss.backward()

        
if __name__ == '__main__':
    
    _is_cuda = torch.cuda.is_available()

    hvd.init()
    rank = hvd.rank()
    f = open('config.json')
    configs =json.load(f)
    torch.manual_seed(configs["MODEL"]["seed"])

    root_dir = configs["ROOT_DATADIR"]["train_dir"]
    _train_dir = root_dir + "train/"
    _label_file_path = os.path.join(root_dir, "train_filepath.csv")
    _train_dataset = CustomDataset(img_dir=_train_dir, label_file_path=_label_file_path)


    allreduce_batch_size = configs["MODEL"]["no_of_batches"] * configs["MODEL"]["batch_size"]
    #custom_sampler = CustomSampler(_train_dataset)
    _train_sampler = dsampler(
            _train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    _train_loader = torch.utils.data.DataLoader(
            _train_dataset, batch_size=allreduce_batch_size,
            sampler=_train_sampler)

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
            _val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    _val_loader = torch.utils.data.DataLoader(
            _train_dataset, batch_size=allreduce_batch_size,
            sampler=_val_sampler)
    '''
    print("test loader:\n")
    it = iter(_val_loader)
    print(next(it))
    '''
    wd = 0.00005
    use_adasum = 0
    MPI.COMM_WORLD.Barrier()
    model = models.resnet50()
    lr_scaler = configs["MODEL"]["batch_size"] * hvd.size() if not use_adasum else 1
    base_lr = 0.0125
    momentum = 0.9


    print ("Finish models") if hvd.rank() == 0 else None
    if _is_cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if use_adasum and hvd.nccl_built():
            lr_scaler = configs["MODEL"]["batch_size"] * hvd.local_size()

    MPI.COMM_WORLD.Barrier()
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=(base_lr * lr_scaler),
                          momentum=momentum, weight_decay=wd)

    MPI.COMM_WORLD.Barrier()

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    epoch_no = 1
    for epoch in range(epoch_no):
        train(epoch)
