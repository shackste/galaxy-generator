""" loss functions
"""
import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss, BCELoss, L1Loss

from parameter import parameter, labels_dim, image_dim

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

mse = MSELoss()
rmse = RMSELoss()
bce = BCELoss()
L1 = L1Loss(reduction="sum")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def get_sample_variance(sample: torch.Tensor) -> torch.Tensor:
    """ calculate variance of sample via MSE from mean"""
    avg = torch.mean(sample, dim=0, keepdims=True)
    sample_variance = mse(sample, avg.repeat(len(sample), *(1,)*(sample.dim()-1)))
    return sample_variance
    
def loss_sample_variance(features: torch.Tensor, threshold=0.001) -> torch.Tensor:
    """ calculate loss of sample variance as mean squared error from average """
    sample_variance = get_sample_variance(features)
    loss = torch.max(torch.tensor(0, device=device), threshold - sample_variance)
    return loss
    
def plot_losses(losses: list, steps: int, iteration: int, label=None) -> None:
#    iterations = list(range(0, iteration+1, steps))
    iterations = torch.arange(len(losses)) * steps
    plt.plot(iterations, losses, label=label)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.xlabel("iteration")



## pytorch has no categorical crossentropy for uncertain onehotencoded target
##   e. g. target = [0.3, 0.2, 0.5] instead of [0,0,1] or actually 2
## use the same loss as tensorflow: - sum target * log(prediction)
def categorical_crossentropy(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """ calculate loss for one hot encoded labels with uncertain target """
    loss = - torch.sum(target * torch.log(prediction+1e-20), dim=-1)
    return loss

cross_entropy = categorical_crossentropy


def loss_reconstruction(image: torch.Tensor, generated_image: torch.Tensor) -> torch.Tensor:
    """ divergence of generated image from input image """
#    return mse(generated_image, image) * image_dim**2
    return L1(generated_image, image)  ## L1 leads to less blurry images, as it penalizes small deviations more strongly


def loss_kl(latent: torch.Tensor) -> torch.Tensor:
    """ divergence of recontstructed latent distribution from true distribution, assumed to be unit gaussian """
    loss = 1 + 2*torch.log(latent[1]) - torch.square(latent[0]) - torch.square(latent[1])
    loss = -0.5 * torch.sum(loss, dim=-1)
    loss = torch.mean(loss)
    return loss


def loss_VAE(image: torch.Tensor, generated_image: torch.Tensor,
             latent_mean: torch.Tensor, latent_std: torch.Tensor) -> torch.Tensor:
    """ total loss of VAE """
    loss = loss_reconstruction(image, generated_image)
    if parameter.alpha:
        loss += loss_kl(latent_mean, latent_std)
    return torch.mean(loss)


def loss_adversarial(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """ divergence of discriminating real and fake images """
    return bce(prediction, target)
#    return mse(prediction, target)  ## MSE leads to more stable training and more qualitative results, 1703.10593

def loss_class(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """ divergence of classification of subclasses in sample distribution """
    loss = cross_entropy(target, prediction)
    loss = torch.mean(loss)
    return loss

def loss_latent(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """ loss for deviation of latent distribution """
    target = torch.cat(target, dim=1)
    prediction = torch.cat(prediction, dim=1)
    return loss_metric(target, prediction)


def loss_metric(target: torch.Tensor, prediction: torch.Tensor) ->  torch.Tensor:
    """ divergence of internal metric """
    return mse(prediction, target)


def loss_generator(target: torch.Tensor, prediction: torch.Tensor,
                   image: torch.Tensor, generated_image: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
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
    return torch.mean(loss)

def loss_discriminator(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """ total loss of discriminator """
    loss = loss_adversarial(target[:,0], prediction[:,0])
    loss += parameter.delta * loss_class(target[:,1:1+labels_dim], prediction[:,1:1+labels_dim])
    return torch.mean(loss)
