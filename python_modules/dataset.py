""" load training data
"""

import numpy as np
from pandas import read_csv
from torch import from_numpy

from file_system import file_galaxy_images, file_galaxy_labels

# galaxy images
def get_x_train():
    x_train = np.load(file_galaxy_images)  ## (N_samples,dim,dim,colors)
    x_train = x_train/255.0 ## rescale to 0<x<1
    x_train = np.rollaxis(x_train, -1, 1)  ## pytorch: (colors,dim,dim)
    x_train = from_numpy(x_train).cuda()
    return x_train

# hierarchical galaxy labels
def get_labels_train():
    df_galaxy_labels =  read_csv(file_galaxy_labels)
    ## for now, only use top level labels
    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:4]].values
    labels_train = from_numpy(labels_train).float().cuda()
    return labels_train
