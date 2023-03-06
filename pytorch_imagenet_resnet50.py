import torch
import horovod.torch as hvd
import argparse, os
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
from utility.sampler import DistributedSampler as dsampler

import json
from utility._dataSplit import split_data
import time
from tqdm import tqdm

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

verbose = 1 if hvd.rank() == 0 else 0

class TimeEstimation(object):
    def __init__(self, name):
        self.name = name
        self.sum_ = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum_ += val
        self.n += 1

    @property
    def avg(self):
        return self.sum_ / self.n

    @property
    def sum(self):
        return self.sum_
    
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


def train(epoch, log_dir):
    torch.cuda.synchronize()
    start_epoch = time.time()

    model.train()
    train_dataset.next_epoch() ## Update the dataset to use the newly received samples
    train_sampler.set_epoch(epoch) ## Set the epoch in sampler and #Create a new indices list
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    rank = hvd.rank()

    if rank ==0:
        init_time_file = open(os.path.join(log_dir, "init_time.log"), "a", buffering=1)
        #shuffling_time_file = open(os.path.join(log_dir, "shuffing.log"), "a", buffering=1)
        io_time_file = open(os.path.join(log_dir, "io_time.log"), "a", buffering=1)
        stagging_file = open(os.path.join(log_dir, "stagging.log"), "a", buffering=1)
        forward_time_file = open(os.path.join(log_dir, "forward_time.log"), "a", buffering=1)
        backward_time_file = open(os.path.join(log_dir, "backward_time.log"), "a", buffering=1)
        weightupdate_time_file = open(os.path.join(log_dir, "weightupdate_time.log"), "a", buffering=1)
        accuracy_file = open(os.path.join(log_dir, "accuracy_per_epoch.log"), "a", buffering=1)
        loss_file = open(os.path.join(log_dir, "loss_per_epoch.log"), "a", buffering=1)
        scheduler_time_file = open(os.path.join(log_dir, "scheduler.log"), "a", buffering=1)
        accuracy_comp_file = open(os.path.join(log_dir, "accuracy_comp_iter.log"), "a", buffering=1)
        #accuracy_iter_file = open(os.path.join(log_dir, "accuracy_per_iter.log"), "a", buffering=1)
        #loss_iter_file = open(os.path.join(log_dir, "loss_per_iter.log"), "a", buffering=1)
        epoch_time_file = open(os.path.join(log_dir, "epoch_time.log"), "a", buffering=1)
        report_time_file =open(os.path.join(log_dir, "report_time.log"), "a", buffering=1)
    MPI.COMM_WORLD.Barrier()

    stop = time.time()
    print("{:.10f}".format(stop - start_epoch), file=init_time_file) if rank == 0 else None
    
    torch.cuda.synchronize()
    start = time.time()
    train_scheduler.scheduling(epoch)
    torch.cuda.synchronize()
    stop = time.time()
    print("SCHD\t{:.10f}".format(stop - start), file=scheduler_time_file) if rank == 0 else None
    
    # print(rank,"START", epoch)
    send_requests, recv_requests = None, None
    with tqdm(total=len(train_loader),
             desc='Train Epoch     #{}'.format(epoch + 1),
             disable=not verbose) as t:
        
        torch.cuda.synchronize()
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            

    if (rank is not 0):
        report_time_file =open(os.path.join(log_dir, "report_time.log"), "a", buffering=1)
    
    io_time = TimeEstimation("io_time")
    stagging_time = TimeEstimation("stagging_time")
    forward_time = TimeEstimation("forward_time")
    backward_time = TimeEstimation("backward_time")
    wu_time = TimeEstimation("wu_time")
    accuracy_comp_time = TimeEstimation("accuracy_comp_time")
    epoch_time = TimeEstimation("epoch_time")
    log_time=TimeEstimation("log_time")

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Is CUDA available? - :" + str(args.cuda))

    hvd.init()
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        print("local rank: {0}", hvd.local_rank())
    rank = hvd.rank()
    print ("Run with arguments:") if hvd.rank() == 0 else None
    for key, value in args._get_kwargs():
        if value is not None:
            print(value,key) if hvd.rank() == 0 else None

    #root_dir = "/home/rongon/Documents/research/shuffling/Codes/ProjectCode/natural_image/data/natural_images/"
    f = open('config.json')
    configs =json.load(f)
    root_dir = configs["ROOT_DATADIR"]["dir"]
    log_dir = configs["ROOT_DATADIR"]["log_dir"]


    #model param configs
    model_configs = configs["MODEL"]
    batch_size = model_configs['batch_size']
    batches_per_allreduce = model_configs['batches_per_allreduce']
    allreduce_batch_size = batch_size * batches_per_allreduce
    use_adasum = 0
    wd = 0.00005


    print("root dir: {0}", root_dir)

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    }

    #Creating own dataset
    #natural_img_dataset = datasets.ImageFolder(
    #                            root = root_dir,
    #                            transform = image_transforms["train"]
    #                    )

    np.random.seed(0)
    torch.manual_seed(0)
    #train_dataset, val_dataset = random_split(natural_img_dataset, (6000, 899))

    sp = split_data()
    sp.split()
    train_folder, val_folder = sp._get_folders()

    train_dataset = datasets.ImageFolder(
                                root = train_folder,
                                transform = image_transforms["train"]
                        )
    
    val_dataset = datasets.ImageFolder(
                                root = val_folder,
                                transform = image_transforms["train"]
                        )

    train_sampler = dsampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler)
    
    val_sampler = dsampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=allreduce_batch_size, sampler=val_sampler)
    
    
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()
    model = models.resnet50()
    lr_scaler = batches_per_allreduce * hvd.size() if not use_adasum else 1

    print ("Finish models") if hvd.rank() == 0 else None
    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if use_adasum and hvd.nccl_built():
            lr_scaler = batches_per_allreduce * hvd.local_size()

    MPI.COMM_WORLD.Barrier()
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=wd)
    
    MPI.COMM_WORLD.Barrier()
    resume_from_epoch = 0
    #if resume_from_epoch > 0 and hvd.rank() == 0:
    #    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    #    checkpoint = torch.load(filepath)
    #    model.load_state_dict(checkpoint['model'])
    #    optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    epoch = 5
    for epoch in range(resume_from_epoch, epoch):
        train(epoch, log_dir )
        validate(epoch, log_dir)