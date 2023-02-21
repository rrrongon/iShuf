import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, random_split



class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end) -> None:
        super(MyIterableDataset).__init__()
        assert end > start, "end must > start"
        self.start = start
        self.end = end
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return iter(range(iter_start, iter_end))
    
#_dataset = MyIterableDataset(10,100)

#for tensor in torch.utils.data.DataLoader(_dataset, num_workers=2):
#    print(tensor)

root_dir = "/home/rongon/Documents/research/shuffling/Codes/ProjectCode/natural_image/data/natural_images/"

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

#Then split the dataset into training and validation set
train_dataset, val_dataset = random_split(natural_img_dataset, (6000, 899))

#for tensor in torch.utils.data.DataLoader(natural_img_dataset, num_workers=1):
#    print(tensor)
#    print(natural_img_dataset.class_to_idx)
#    break

