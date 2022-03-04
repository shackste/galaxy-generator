import numpy as np
import torch
from geomloss import SamplesLoss


def wasserstein(a, b,
                blur: float = 0.05, # smaller seem better, and possibly slower?
                scaling: float = 0.8, # the closer to 1 the slower but more accurate.
                splits: int = 1 # set as small as possible so that data fits on GPU.
                ):
    """ Compute the Wasserstein distance between two sets of features, a and b, of arbitrary size. """
    # separate data into portions that can be handled by the GPU
    a, b = [np.split(x, splits) for x in [a, b]]
    # initiate loss function
    Loss = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, backend="tensorized")
    # compute distance
    distance = np.mean([Loss(x, y).item() for x in loop_cuda(a) for y in loop_cuda(b)])
    return distance


def loop_numpy2cuda(iterable):
    """ loop over iterable and transform every element from numpy to torch.tensor and put on cuda """
    for x in iterable:
        yield torch.from_numpy(x).cuda()

def loop_cuda(iterable):
    """ loop over iterable and transform every element from numpy to torch.tensor and put on cuda """
    for x in iterable:
        yield x.cuda()