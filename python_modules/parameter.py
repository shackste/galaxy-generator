""" hardcoded parameter

these can be changed in a jupyter notebook during runtime via

>>> import parameter
>>> parameter.parameter = new_value

"""

from torch.optim import Adam

# Input
image_dim = 64
colors_dim = 3
labels_dim = 3
input_size = (colors_dim,image_dim,image_dim)

# Encoder/Decoder
latent_dim = 8
decoder_dim = latent_dim  # differs from latent_dim if PCA applied before decoder

# General
learning_rate = 0.0002
betas = (0.5,0.999)  ## 0.999 is default beta2 in tensorflow
optimizer = Adam
negative_slope = 0.2 # for LeakyReLU
momentum = 0.99 # for BatchNorm

# Loss weights
alpha = 1 # switch VAE (1) / AE (0)
beta = 1 # weight for KL-loss
zeta = 0.5 # weight for MSE-loss
delta = 1 # weight for class-loss
gamma = 1024 # weight for learned-metric-loss (https://arxiv.org/pdf/1512.09300.pdf)
