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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--test-arg', default='',
                    help='test passed argument')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Is CUDA available? - :" + str(args.cuda))

    hvd.init()
    rank = hvd.rank()
    print ("Run with arguments:") if hvd.rank() == 0 else None
    for key, value in args._get_kwargs():
        if value is not None:
            print(value,key) if hvd.rank() == 0 else None

    #root_dir = "/home/rongon/Documents/research/shuffling/Codes/ProjectCode/natural_image/data/natural_images/"
    f = open('config.json')
    configs =json.load(f)
    root_dir = configs["ROOT_DATADIR"]["dir"]

    print("root dir: {0}", root_dir)

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    }

    #Creating own dataset
    natural_img_dataset = datasets.ImageFolder(
                                root = root_dir,
                                transform = image_transforms["train"]
                        )

    np.random.seed(0)
    torch.manual_seed(0)
    #train_dataset, val_dataset = random_split(natural_img_dataset, (6000, 899))

    print("from rank: {0}: {1}", rank, natural_img_dataset.class_to_idx)