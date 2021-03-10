""" sample functions
"""

from torch import randn

# latent space
def gaussian_sampler(mu, sigma):
    batch = mu.shape[0]
    dim = mu.shape[1]
    return mu + sigma*randn(batch, dim, requires_grad=True).cuda()


def return_batch(sample, i, size):
    """ from sample return the i'th batch of size """
    return sample[i*size : (i+1)*size]
