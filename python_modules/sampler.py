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


######################
## Training sampler ##
######################


## completely random batch each next()
def make_training_sample_generator(batch_size, x_train, labels_train):
    while True:
        idx = np.random.choice(range(N_samples), size=batch_size, replace=False)
        yield x_train[idx], labels_train[idx]


## randomized batches with each sample chosen at most once per epoch
def make_training_sample_generator(batch_size, x_train, labels_train):
    indices = np.random.permutation(N_samples)
    for i in range(int(N_samples/batch_size)):
        idx = return_batch(indices, i, batch_size)
        yield x_train[idx], labels_train[idx]
