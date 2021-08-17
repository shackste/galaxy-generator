""" hardcoded parameter

these can be changed in a jupyter notebook during runtime via

>>> import parameter
>>> parameter.parameter = new_value

"""

from torch.optim import Adam

###############
## hardcoded ##
###############


# Input
image_dim = 64
colors_dim = 3
labels_dim = 37 #3
input_size = (colors_dim,image_dim,image_dim)


#############
## mutable ##
#############

class Parameter:
    """ container for hyperparameters"""

    def __init__(self):
        # Encoder/Decoder
        self.latent_dim = 8
        self.decoder_dim = self.latent_dim  # differs from latent_dim if PCA applied before decoder

        # General
        self.learning_rate = 0.0002
        self.betas = (0.5,0.999)  ## 0.999 is default beta2 in tensorflow
        self.optimizer = Adam
        self.negative_slope = 0.2 # for LeakyReLU
        self.momentum = 0.99 # for BatchNorm

        # Loss weights
        self.alpha = 1 # switch VAE (1) / AE (0)
        self.beta = 1 # weight for KL-loss
        self.gamma = 1024 # weight for learned-metric-loss (https://arxiv.org/pdf/1512.09300.pdf)
        self.delta = 1 # weight for class-loss
        self.zeta = 0.5 # weight for MSE-loss

    def return_parameter_dict(self):
        return(self.__dict__)

parameter = Parameter()
