import torch
import horovod.torch as hvd
import argparse, os
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import json
import random
import shutil

THRESH_HOLD = 5
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def copy_dataset_class(in_folder, out_folder,filenames): #, f):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    for filename in filenames:
        source_file_path = os.path.join(in_folder,filename)
        dest_file_path = os.path.join(out_folder,filename)
        command = "cp " + str(source_file_path) + " " + str(dest_file_path)
        #print("command: {0}".format(command))
        #print("Source file: {0}\n dest file: {1}".format(source_file_path, dest_file_path))
        os.system(command)

class split_data:
    def __init__(self, val_parcent=20):
        self.val_parcent = val_parcent

        f = open('config.json')
        configs =json.load(f)
        #self.root_dir = configs["ROOT_DATADIR"]["dir"] #Partition
        self.root_dir = configs["ROOT_DATADIR"]["partiion_dir"]
        #self.source_data_dir = configs["ROOT_DATADIR"]["data_dir"] # Folder name by rank number
        self.source_data_dir = self.root_dir + str(rank)

        folder_name = str(rank)+"_data_2"
        self.target_folder = os.path.join(self.root_dir, folder_name) # target folder must be inside the rank folder that is self.source_data_dir
        if not os.path.exists(self.target_folder):
            os.mkdir(self.target_folder)
        self.train_data_dir = os.path.join(self.target_folder, 'train')
        self.val_data_dir = os.path.join(self.target_folder, 'val')

        if not os.path.exists(self.train_data_dir):
            os.mkdir(self.train_data_dir)
        if not os.path.exists(self.val_data_dir):
            os.mkdir(self.val_data_dir)
    
    def split(self):
        classes = []
        imgs = []
        total_samples = 0

        all_classes = os.listdir(self.source_data_dir)
        print(len(all_classes))

        all_classes.sort()
        for _class in all_classes:
            d = os.path.join(self.source_data_dir, _class)
            if not os.path.isdir(d):
                print(d,"is not dir")
                continue

            filenames = []
            for filename in os.listdir(d):
                if is_image_file(filename):
                    path = '{0}/{1}'.format(_class, filename)
                    item = (path, _class)
                    #filenames.append(item)
                    filenames.append(path)
                
            if len(filenames) >= THRESH_HOLD:
                total_samples = total_samples + len(filenames)
                classes.append((_class,len(filenames)))
                random.seed(230)
                filenames.sort()
                random.shuffle(filenames)
                split = int( (1-self.val_parcent*0.01) * len(filenames))
                train_filenames = filenames[:split]
                val_filenames = filenames[split:]

                #print("train file len {0} and val file len {1}".format(len(train_filenames), len(val_filenames)))
                try:
                    train_sub_folder = os.path.join(self.train_data_dir, _class)
                    if not os.path.exists(train_sub_folder):
                        os.mkdir(train_sub_folder)
                    val_sub_folder =  os.path.join(self.val_data_dir, _class)
                    if not os.path.exists(val_sub_folder):
                        os.mkdir(val_sub_folder)

                    copy_dataset_class(self.source_data_dir,self.train_data_dir, train_filenames)#,f1)
                    # for filename in train_filenames:
                    # f1.write(str(filename) + "\n")
                    copy_dataset_class(self.source_data_dir,self.val_data_dir, val_filenames) #,f2)
                    
                except:
                    print("len of traing files: {0}, val files{1}".format(len(train_filenames), len(val_filenames)))
                    print("Exception with class",_class)
                    shutil.rmtree(self.data_folder)
            else:
                print("Remove class",_class, "with",len(filenames),"samples")

    def _get_folders(self):
        return self.train_data_dir, self.val_data_dir
if __name__ == '__main__':

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    sp = split_data()
    sp.split()
    sp._get_folders()
