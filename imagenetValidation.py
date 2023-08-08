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

class ImageNetValidationDataset(Dataset):
    def __init__(self, ROOT, validation_file, class_labels_file, CLASS_NUMBER, transform=None):
        self.root_folder = ROOT
        self.validation_file = validation_file
        self.class_labels_file = class_labels_file
        self.transform = transforms.Compose([
             transforms.Resize(256),                          # Resize the image to 256x256 pixels
             transforms.CenterCrop(224),                      # Crop the center 224x224 pixels
             transforms.Grayscale(num_output_channels=3),     # Convert the image to RGB if it's grayscale
             transforms.ToTensor(),                            # Convert the image to a PyTorch tensor
             transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image channels
                                 std=[0.229, 0.224, 0.225])
         ])
        self.num_classes = CLASS_NUMBER

        # Read wnids and image paths from validation file
        with open(self.validation_file, 'r') as file:
            self.data = [line.strip().split() for line in file]

        # Read class labels for each wnid from class_labels file and create a mapping
        self.wnid_to_label = {}
        with open(self.class_labels_file, 'r') as file:
            for line in file:
                wnid, label = line.strip().split()
                self.wnid_to_label[wnid] = int(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, wnid = self.data[index]
        image_path = os.path.join(self.root_folder, image_path)
        # Open and load the image using PIL
        #image = Image.open(image_path).convert('RGB')

        image = PIL.Image.open(image_path)
        # Apply the transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        # Get the label corresponding to the wnid
        label = self.wnid_to_label[wnid]

        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[label] = 1
        return image, y_one_hot
