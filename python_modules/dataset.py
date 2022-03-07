""" load training data
"""

import numpy as np
from pandas import read_csv
import  torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, RandomAffine, Resize, RandomVerticalFlip, RandomHorizontalFlip, Normalize
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
from torch.utils.data.dataloader import default_collate

from file_system import file_galaxy_images, file_galaxy_labels, folder_images

# galaxy images
def get_x_train() -> torch.Tensor:
    x_train = np.load(file_galaxy_images)  ## (N_samples,dim,dim,colors)
    x_train = x_train/255.0 ## rescale to 0<x<1
    x_train = np.rollaxis(x_train, -1, 1)  ## pytorch: (colors,dim,dim)
    x_train = torch.from_numpy(x_train)
    return x_train

# hierarchical galaxy labels
def get_labels_train() -> torch.Tensor:
    df_galaxy_labels =  read_csv(file_galaxy_labels)
    ## for now, only use top level labels
    ##    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:4]].values
    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:]].values
    labels_train = torch.from_numpy(labels_train).float()
    return labels_train

## Dataset

class DataSet(Dataset):
    def __init__(self):
        self.path_images = folder_images
        file_list = glob(self.path_images + "*")
        file_list.sort()
        labels = get_labels_train() ## labels are sorted already
        self.test_data = False
        self.data = []
        for file, label in zip(file_list, labels):
            self.data.append([file, label])

        self.augment = Compose([
#            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomAffine((0, 360), (0.01,) * 2),  # rotation, -4 to 4 translation
            CenterCrop(207),
            #    Resize((150,)*2),
            #    Resize((100,)*2),
            Resize((64,) * 2),
            ToTensor(),
            Normalize(0.5, 0.5),
        ])

        self.augment_test = Compose([
            #    RandomVerticalFlip(),
            #    RandomHorizontalFlip(),
            #    RandomAffine((0,360), (0.01,)*2),   # rotation, -4 to 4 translation
            CenterCrop(207),
            #    Resize((150,)*2),
            #    Resize((100,)*2),
            Resize((64,) * 2),
            ToTensor(),
            Normalize(0.5, 0.5),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img = Image.open(file)
        if self.test_data:
            img = self.augment_test(img)
        else:
            img = self.augment(img)
        return img, label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class MakeDataLoader:
    def __init__(self, test_size=0.1, random_state=2, augmented=True):
        self.dataset = DataSet()
        if not augmented:
            self.dataset.test_data = True
        train_idx, test_idx = train_test_split(list(range(len(self.dataset))), test_size=test_size, random_state=random_state)
        valid_idx, test_idx = train_test_split(list(range(len(test_idx))), test_size=0.5, random_state=random_state+1)

        self.dataset_train = Subset(self.dataset, np.array(train_idx))
        self.dataset_valid = Subset(self.dataset, np.array(valid_idx))
        self.dataset_test = Subset(self.dataset, np.array(test_idx))
        self.dataset_test.test_data = True
        self.dataset_train.test_data = True if not augmented else False
        self.dataset_valid.test_data = True if not augmented else False
        self.dataset.test_data = True if not augmented else False

    def get_data_loader_full(self, batch_size=64, shuffle=True, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=device.type=="gpu", **kwargs)

    def get_data_loader_train(self, batch_size=64, shuffle=True, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=device.type=="gpu", **kwargs)

    def get_data_loader_test(self, batch_size=64, shuffle=False, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_test, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=device.type=="gpu", **kwargs)

    def get_data_loader_valid(self, batch_size=64, shuffle=False, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=device.type=="gpu", **kwargs)
