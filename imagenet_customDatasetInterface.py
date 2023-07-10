import os
import torch
from PIL import Image
import warnings
warnings.simplefilter("ignore", UserWarning)

from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import os, json
import pandas as pd
import horovod.torch as hvd
from  torchvision import transforms, models
import PIL
from mpi4py import MPI
import sys

def rank_replace_img_path(rank, old_path):
    # split the path into its components
    path_components = old_path.split("/")
    path_components[-4] = 'parition'+str(rank)
    new_path = os.path.join(*path_components)
    #new_path = "/"+new_path
    #print("Old path#{0} and new path#{1}".format(old_path, new_path))
    return new_path

class ImageNetDataset(Dataset):
    def __init__(self, data_folder, wnids_file, words_file, DATASET_TYPE, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.wnid_label_dic = dict()
        self.classes, self.class_labels = self.read_classes(wnids_file, words_file)
        self.image_paths, self.labels = self.collect_image_paths()
        self.transform = transforms.Compose([
            transforms.Resize(256),                          # Resize the image to 256x256 pixels
            transforms.CenterCrop(224),                      # Crop the center 224x224 pixels
            transforms.Grayscale(num_output_channels=3),     # Convert the image to RGB if it's grayscale
            transforms.ToTensor(),                            # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image channels
                                std=[0.229, 0.224, 0.225])
        ])
        self.num_classes = DATASET_TYPE


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        #image = Image.open(image_path).convert('RGB')

        image = PIL.Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[label] = 1

        return image, y_one_hot, image_path, index

    def read_classes(self, wnids_file, words_file):
        with open(wnids_file, 'r') as file:
            wnids = [line.strip() for line in file]

        with open(words_file, 'r') as file:
            class_names = [line.strip() for line in file]

        return class_names, wnids

    def collect_image_paths(self):
        image_paths = []
        labels = []
        for class_label, wnid in enumerate(self.class_labels):
            class_folder = os.path.join(self.data_folder, wnid)
            try:
                image_files = os.listdir(class_folder)

                for image_file in image_files:
                    image_path = os.path.join(class_folder, image_file)
                    image_paths.append(image_path)
                    labels.append(class_label)
            except Exception as e:
                print(f"issue on class folder{class_folder} and exception {e}")
            self.wnid_label_dic[class_label] = wnid
        return image_paths, labels

    def add_new_samples(self, rank, recvd_samples):
        try:
            for sample in recvd_samples:
                path = sample['path']
                label = torch.argmax(sample['label']).item()
                #label = sample['label']
                #image = sample['sample']

                replaced_path = rank_replace_img_path(rank, path) #when multiple ranks
                #if rank ==0:
                #    print("received sample path: {0}.\n Replaced path:{1}".format(path, replaced_path))
                #    sys.stdout.flush()

                self.image_paths.append(replaced_path)
                self.labels.append(label)
        except Exception as e:
            print("Error in adding new samples in rank#{0}".format(rank))
            print("Exception on Rank#{0}".format(str(e)))
            sys.stdout.flush()
            hvd.Abort()

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

    def remove_old_samples(self, rank, clean_list):
        filtered_paths = [p for i, p in enumerate(self.image_paths) if i not in clean_list]
        filtered_labels = [l for i, l in enumerate(self.labels) if i not in clean_list]

        for i in clean_list:
            file_path = self.image_paths[i]
            os.remove(file_path)

        self.image_paths = filtered_paths
        self.labels = filtered_labels


    def get_imagepaths(self):
        return self.image_paths

    def get_labels(self):
        return self.labels
'''
train_folder = '/home/rongon/Documents/research/shuffling/Codes/ProjectCode/datasets/imagenet-mini/train'
val_folder = '/home/rongon/Documents/research/shuffling/Codes/ProjectCode/datasets/imagenet-mini/val'

wnids_file = '/home/rongon/Documents/research/shuffling/Codes/ProjectCode/datasets/imagenet_mini/wnids.txt'
words_file = '/home/rongon/Documents/research/shuffling/Codes/ProjectCode/datasets/imagenet_mini/words.txt'
transform = ...  # define the image transformations as needed

train_dataset = ImageNetDataset(train_folder, wnids_file, words_file, transform=None)
val_dataset = ImageNetDataset(val_folder, wnids_file, words_file, transform=None)
'''


