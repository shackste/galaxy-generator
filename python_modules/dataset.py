""" load training data
"""

import numpy as np
from pandas import read_csv
from torch import from_numpy
from torchvision.transforms import Compose, CenterCrop, ToTensor, RandomAffine, Resize, RandomVerticalFlip, RandomHorizontalFlip, RandomErasing, ToPILImage
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image


from file_system import file_galaxy_images, file_galaxy_labels, folder_images

# galaxy images
def get_x_train():
    x_train = np.load(file_galaxy_images)  ## (N_samples,dim,dim,colors)
    x_train = x_train/255.0 ## rescale to 0<x<1
    x_train = np.rollaxis(x_train, -1, 1)  ## pytorch: (colors,dim,dim)
    x_train = from_numpy(x_train)
    return x_train

# hierarchical galaxy labels
def get_labels_train():
    df_galaxy_labels =  read_csv(file_galaxy_labels)
    ## for now, only use top level labels
    ##    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:4]].values
    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:]].values
    labels_train = from_numpy(labels_train).float()
    return labels_train


## augmentation

augment = Compose([
    RandomVerticalFlip(),
    RandomHorizontalFlip(),
    RandomAffine((0,360), (0.01,)*2),   # rotation, -4 to 4 translation
    CenterCrop(207),
#    Resize((150,)*2),
#    Resize((100,)*2),
    Resize((64,)*2),
    ToTensor(),
    RandomErasing(scale=(0.01, 0.2)),
])


## Dataset

class DataSet(Dataset):
    def __init__(self):
        self.path_images = folder_images
        file_list = glob(self.path_images + "*")
        labels = get_labels_train()
        self.data = []
        for file, label in zip(file_list, labels):
            self.data.append([file, label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img = Image.open(file)
        img = augment(img)
#        img = plt.imread(file, format='jpg')
#        img = tensor(img).permute(2,0,1)
        return img, label

class MakeDataLoader:
    def __init__(self, test_size=0.1, random_state=2):
        dataset = DataSet()
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_size, random_state=random_state)
        self.dataset_train = Subset(dataset, train_idx)
        self.dataset_test = Subset(dataset, test_idx)

    def get_data_loader_train(self, batch_size=64):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, drop_last=True )

    def get_data_loader_test(self, batch_size=64):
        return DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True, drop_last=True )