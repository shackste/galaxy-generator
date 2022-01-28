"""
This file contains the Variational Autoencoder (VAE) used to reduce galaxy images to a feature space,
where the distribution of features can be compared between diffirent sets of images.
"""
from tqdm import trange

import wandb
from torch.optim import Adam
from torch.nn import Sequential, ModuleList, \
                     Conv2d, Linear, \
                     LeakyReLU, Softplus, Tanh, \
                     BatchNorm1d, BatchNorm2d, Flatten, \
                     ConvTranspose2d, UpsamplingBilinear2d

from helpful_functions import write_generated_galaxy_images_iteration
from sampler import gaussian_sampler
from loss import loss_reconstruction,loss_kl
from dataset import MakeDataLoader
from neuralnetwork import NeuralNetwork

# parameters for VAE
colors_dim = 3
momentum = 0.99 # Batchnorm
negative_slope = 0.2 # LeakyReLU
optimizer = Adam
betas = (0.5, 0.999)

# hyperparameters
learning_rate = 2e-4
latent_dim = 16
alpha = 0.0005

# wandb
hyperparameters = {
    "learning_rate": learning_rate,
    "dim_z": latent_dim,
    "weight_loss_kl": alpha,
}
wandb_kwargs = {
    "project" : "galaxy generator",  # top level identifier
    "group" : "reduction VAE",  # secondary identifier
    "job_type" : "training",  # third level identifier
    "tags" : ["training", "parameter search"],  # tags for organizing tasks
    "name" : f"lr {learning_rate}, dim_z {latent_dim}, alpha {alpha}", # bottom level identifier, label of graph in UI
    "config" : hyperparameters, # dictionary of used hyperparameters
}


def train_autoencoder(epochs: int = 2000, batch_size: int = 64, num_workers=4, track=False, plot_images: int = 0):
    """ perform training loop for the VAE """
    if track:
        wandb.login(key="834835ffb309d5b1618c537d20d23794b271a208")
        wandb.init(**wandb_kwargs)

    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    make_data_loader = MakeDataLoader(augmented=True)
    data_loader = make_data_loader.get_data_loader_train(batch_size=batch_size, 
                                                         shuffle=True,
                                                         num_workers=num_workers)
    for epoch in trange(epochs, desc="epochs"):
        for images, _ in data_loader:
            images = images.cuda()
            images = images*2 - 1 # rescale (0,1) to (-1,1)

            latent_mu, latent_sigma = encoder(images)
            latent = gaussian_sampler(latent_mu, latent_sigma) #.unsqueeze(2).unsqueeze(3)
            generated_images = decoder(latent)

            decoder.zero_grad()
            encoder.zero_grad()
            loss_recon_ = loss_reconstruction(images, generated_images)
            loss_kl_ = alpha*loss_kl([latent_mu, latent_sigma])
            loss = loss_recon_ + loss_kl_
            loss.backward()

            encoder.optimizer.step()
            decoder.optimizer.step()
        encoder.save()
        decoder.save()
        if track:
            log = {
                "loss reconstruction": loss_recon_.item(),
                "loss KL": loss_kl_.item(),
                "loss": loss.item()
            }
            wandb.log(log)
        if plot_images and not epoch % plot_images:
            width = min(8, len(generated_images))
            write_generated_galaxy_images_iteration(iteration=epoch, images=generated_images, width=width,
                                                    height=len(generated_images) // width, file_prefix="generated_VAEreduc")
    wandb.finish()


class Encoder(NeuralNetwork):
    """ convolutional network with BatchNorm and LeakyReLU """
    def __init__(self, dim_z=latent_dim):
        super(Encoder, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size)
        self.dim_z = dim_z

        self.conv0 = Sequential(
            Conv2d(colors_dim, 16, kernel_size=1, stride=1),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.conv1 = Sequential(
            Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(32, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.conv2 = Sequential(
            Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(64, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.conv3 = Sequential(
            Conv2d(64,128, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(128, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
            Flatten(), # next layer takes flat input with labels appended
        )
        self.dense1 = Sequential(
            Linear(8192, 2048),
            BatchNorm1d(2048, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope)
        )

        ## the following take the same input from dense1
        self.dense_z_mu = Linear(2048, self.dim_z)
        self.dense_z_std = Sequential(
            Linear(2048, self.dim_z),
            Softplus(),
        )
        self.set_optimizer(optimizer, lr=learning_rate, betas=betas)

    def forward(self, images):
        x = self.conv0(images)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        z_mu = self.dense_z_mu(x)
        z_std = self.dense_z_std(x)
        return z_mu, z_std


class Decoder(NeuralNetwork):
    def __init__(self, ll_scaling=1.0, dim_z=latent_dim):
        super(Decoder, self).__init__()
        self.dim_z = dim_z
        ngf = 32
        self.init = genUpsample(self.dim_z,ngf * 16,1, 0)
        self.m_modules = ModuleList()#to 4x4
        self.c_modules  = ModuleList()
        for i in range(4):
          self.m_modules.append(genUpsample2(ngf * 2**(4-i), ngf * 2**(3-i),3))
          self.c_modules.append(Sequential(Conv2d(ngf * 2**(3-i), colors_dim, 3, 1, 1, bias=False), Tanh()))
        self.set_optimizer(optimizer, lr=learning_rate*ll_scaling, betas=betas)

    def forward(self, latent, step=3, alpha=1):
        out = self.init(latent.unsqueeze(2).unsqueeze(3))
        for i in range (step):
            out =  self.m_modules[i](out)
        out2 = self.c_modules[step](self.m_modules[step](out))
        if alpha == 1 or not step:   return out2
        out = F.interpolate(self.c_modules[step-1](out), scale_factor=2, mode='bilinear', align_corners=False)
        return (1-alpha)*out + (alpha)*out2

def genUpsample(input_channels, output_channels, stride, pad):
   return Sequential(
        ConvTranspose2d(input_channels, output_channels, 4, stride, pad, bias=False),
        BatchNorm2d(output_channels),
        LeakyReLU(negative_slope=negative_slope))

def genUpsample2(input_channels, output_channels, kernel_size):
   return Sequential( 
        Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding= (kernel_size-1) // 2 ),
        BatchNorm2d(output_channels),
        LeakyReLU(negative_slope=negative_slope),
        Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding= (kernel_size-1) // 2 ),
        BatchNorm2d(output_channels),
        LeakyReLU(negative_slope=negative_slope),
        UpsamplingBilinear2d(scale_factor=2))

