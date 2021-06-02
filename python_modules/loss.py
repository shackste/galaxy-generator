""" loss functions
"""

from torch import sum, log, mean, square
from torch.nn import MSELoss, BCELoss, L1Loss

from parameter import parameter, labels_dim, image_dim

mse = MSELoss()
bce = BCELoss()
L1 = L1Loss(reduction="sum")


## pytorch has no categorical crossentropy for uncertain onehotencoded target
##   e. g. target = [0.3, 0.2, 0.5] instead of [0,0,1] or actually 2
## use the same loss as tensorflow: - sum target * log(prediction)
def categorical_crossentropy(target, prediction):
    """ calculate loss for one hot encoded labels with uncertain target """
    loss = - sum(target * log(prediction+1e-20), dim=-1)
    return loss

cross_entropy = categorical_crossentropy


def loss_reconstruction(image, generated_image):
    """ divergence of generated image from input image """
#    return mse(generated_image, image) * image_dim**2
    return L1(generated_image, image)  ## L1 leads to less blurry images, as it penalizes small deviations more strongly


def loss_kl(latent):
    """ divergence of recontstructed latent distribution from true distribution, assumed to be unit gaussian """
    loss = 1 + 2*log(latent[1]) - square(latent[0]) - square(latent[1])
    loss = -0.5 * sum(loss, dim=-1)
    loss = mean(loss)
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
#    return mse(prediction, target)  ## MSE leads to more stable training and more qualitative results, 1703.10593

def loss_class(target, prediction):
    """ divergence of classification of subclasses in sample distribution """
    loss = cross_entropy(target, prediction)
    loss = mean(loss)
    return loss

def loss_latent(target, prediction):
    """ loss for deviation of latent distribution """
    target = cat(target, dim=1)
    prediction = cat(prediction, dim=1)
    return loss_metric(target, prediction)


def loss_metric(target, prediction):
    """ divergence of internal metric """
    return mse(prediction, target)


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
