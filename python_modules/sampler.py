""" sample functions
"""

import types
import numpy as np
import torch
from torch import randn, rand, sqrt
from torch.nn import Softmax

from parameter import parameter
from labeling import make_galaxy_labels_hierarchical, labels_dim, label_group_sizes

# latent space
def gaussian_sampler(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    batch = mu.shape[0]
    dim = mu.shape[1]
    return mu + sigma*torch.randn(batch, dim, requires_grad=True).cuda()

def generate_latent(batch_size: int) -> torch.Tensor:
    """ generate latent distribution """
    latent_mu = torch.randn(batch_size, parameter.latent_dim).cuda()
    latent_sigma = torch.rand(batch_size, parameter.latent_dim).cuda()
    return latent_mu, latent_sigma


# labels


def generate_galaxy_labels(batch_size: int) -> torch.Tensor:
    """ generate a batch of hierarchical galaxy labels """
    norm = torch.nn.Softmax(dim=1)
    groups = [norm(rand(batch_size, l)) for l in label_group_sizes]
    groups = make_galaxy_labels_hierarchical(groups)
    return groups.detach().cuda()


def generate_noise(batch_size: int) -> torch.Tensor:
    """ generate random noise """
    noise = torch.rand(batch_size, labels_dim).cuda()
    return noise


def return_batch(sample, i, size):
    """ from sample return the i'th batch of size """
    return sample[i*size : (i+1)*size]


######################
## Training sampler ##
######################


## completely random batch each next()
def make_training_sample_generator(batch_size: int, x_train: np.array, labels_train: np.array) -> types.GeneratorType:
    N_samples = x_train.shape[0]
    while True:
        idx = np.random.choice(range(N_samples), size=batch_size, replace=False)
        yield x_train[idx].cuda(), labels_train[idx].cuda()


## randomized batches with each sample chosen at most once per epoch
def make_training_sample_generator(batch_size: int, x_train: np.array, labels_train: np.array) -> types.GeneratorType:
    N_samples = x_train.shape[0]
    indices = np.random.permutation(N_samples)
    for i in range(int(N_samples/batch_size)):
        idx = return_batch(indices, i, batch_size)
        yield x_train[idx].cuda(), labels_train[idx].cuda()
