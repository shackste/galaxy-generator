import torch
from torch.nn import Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, Linear, Sigmoid, Tanh, Flatten, BCELoss

from neuralnetwork import NeuralNetwork
from additional_layers import PrintShape

optimizer = torch.optim.Adam

class Discriminator(NeuralNetwork):
    def __init__(self, labels_dim=0, D_lr=2e-4):
        super(Discriminator, self).__init__()

        self.conv = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2, inplace=True),
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(32),
            LeakyReLU(0.2, inplace=True),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
        )
        self.flatten = Flatten()
        self.linear = Sequential(
            Linear(256, 1),
            Tanh(),
        )

        self.set_optimizer(optimizer, lr=D_lr)

    def forward(self, images, labels):
        x = self.conv(images)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Generator(NeuralNetwork):
    def __init__(self, latent_dim, labels_dim, G_lr=5e-5):
        super(Generator, self).__init__()

        self.embed = Linear(labels_dim, latent_dim)

        self.linear = Linear(latent_dim*2, 128*4*4)

        self.conv = Sequential(
            ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(32),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(16),
            LeakyReLU(0.2, inplace=True),
            ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=2),
            Tanh(),
        )

        self.set_optimizer(optimizer, lr=G_lr)


    def forward(self, latent, labels):
        x = torch.cat([latent, self.embed(labels)], dim=1)
        x = self.linear(x)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.conv(x)
        return x


bce = BCELoss()


def discriminator_loss(prediction_fake, prediction_real):
    fake = torch.ones_like(prediction_fake) * 0.1
    real = torch.ones_like(prediction_real) * 0.9
    loss_real = bce(0.5*(prediction_real+1), real)
    loss_fake = bce(0.5*(prediction_fake+1), fake)
    return loss_real, loss_fake


def generator_loss(prediction):
    target = torch.ones_like(prediction) * 0.9
    loss = bce(0.5*(prediction+1), target)
    return loss