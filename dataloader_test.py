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
from utils.sampler import DistributedSampler as dsampler
import horovod.torch as hvd

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
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self._labels.iloc[idx,0])
        image = read_image(img_path)
        label = self._labels.iloc[idx, 1]
        new_label = str(label) + "_" + str(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, new_label

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

    print("train loader:\n")
    it = iter(_train_loader)
    print(next(it))

    _val_dir = root_dir + "val/"
    _label_file_path = os.path.join(root_dir, "val_filepath.csv")
    _val_dataset = CustomDataset(img_dir=_val_dir, label_file_path=_label_file_path)

    #custom_sampler = CustomSampler(_val_dataset)
    _val_sampler = dsampler(
            _val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    _val_loader = torch.utils.data.DataLoader(
            _train_dataset, batch_size=allreduce_batch_size,
            sampler=_val_sampler)

    print("test loader:\n")
    it = iter(_val_loader)
    print(next(it))