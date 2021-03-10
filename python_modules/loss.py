""" loss functions
"""

from torch import sum, log, mean
from torch.nn import MSELoss, BCELoss

from parameter import parameter, labels_dim, image_dim

mse = MSELoss()
bce = BCELoss()


## pytorch has no categorical crossentropy for uncertain onehotencoded target
##   e. g. target = [0.3, 0.2, 0.5] instead of [0,0,1] or actually 2
## use the same loss as tensorflow: - sum target * log(prediction)
def categorical_crossentropy(target, prediction):
    """ calculate loss for one hot encoded labels with uncertain target """
    loss = - sum(target * log(prediction))
    return loss

cross_entropy = categorical_crossentropy


def loss_reconstruction(image, generated_image):
    """ divergence of generated image from input image """
    return mse(image, generated_image) * image_dim**2


def loss_kl(latent):
    """ divergence of recontstructed latent distribution from true distribution, assumed to be unit gaussian """
    loss = 1 + 2*log(latent[1]) - square(latent[0]) - square(latent[1])
    loss = -0.5 * sum(loss) #, axis=-1)
    return loss


def loss_VAE(image, generated_image, latent_mean, latent_std):
    """ total loss of VAE """
    loss = loss_reconstruction(image, generated_image)
    if parameter.alpha:
        loss += loss_kl(latent_mean, latent_std)
    return mean(loss)


def loss_adversarial(target, prediction):
    """ divergence of discriminating real and fake images """
    return bce(prediction, target)


def loss_class(target, prediction):
    """ divergence of classification of subclasses in sample distribution """
    return cross_entropy(target, prediction)


def loss_metric(target, prediction):
    """ divergence of internal metric """
    return mse(target, prediction)


def loss_generator(target, prediction, image, generated_image, latent):
    """ total loss of generator
    
    target and prediction contain (axis=1)
        0               ; binary classification
        1:labels_dim    ; label classification
        labels_dim+1:-1 ; metric
    """
    loss = loss_adversarial(target[:,0], prediction[:,0])
    loss += parameter.delta * loss_class(target[:,1:1+labels_dim], prediction[:,1:1+labels_dim])
    loss += parameter.gamma * loss_metric(target[:,2+labels_dim:], prediction[:,2+labels_dim:])
    loss += parameter.zeta * loss_reconstruction(image, generated_image)
    if parameter.alpha:
        loss += parameter.beta * loss_kl(latent)
    return mean(loss)

def loss_discriminator(target, prediction):
    """ total loss of discriminator """
    loss = loss_adversarial(target[:,0], prediction[:,0])
    loss += parameter.delta * loss_class(target[:,1:1+labels_dim], prediction[:,1:1+labels_dim])
    return mean(loss)
