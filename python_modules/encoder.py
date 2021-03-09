""" Encoder networks for RGB images """

from torch import cat
from torch.nn import Sequential, \
                     Conv2d, Linear, \
                     ReLU, LeakyReLU, Sigmoid, Softmax, Softplus, \
                     BatchNorm1d, BatchNorm2d, Flatten


from neuralnetwork import NeuralNetwork
from parameter import colors_dim, latent_dim, labels_dim, \
                      optimizer, learning_rate, betas, \
                      momentum, negative_slope, \
                      alpha



class Encoder1(NeuralNetwork):
    def __init__(self):
        super(Encoder1, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size)
        self.conv1 = Sequential(
            Conv2d(colors_dim, 8, kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU(),
        )
        self.conv2 = Sequential(
            Conv2d(8, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(16),
            ReLU(),
        )
        self.conv3 = Sequential(
            Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU(),
        )
        self.conv4 = Sequential(
            Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU(),
        )
        self.dense1 = Sequential(
            Flatten(),
            Linear(1024, 256),
            ReLU(),
        )
        
        ## the following take the same input
        self.dense_z_mu = Linear(256, latent_dim)
        if alpha:
            self.dense_z_std = Sequential(
                Linear(256, latent_dim),
                Softplus(),
            )
        self.set_optimizer(optimizer, lr=learning_rate, betas=betas)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dense1(x)
        z_mu = self.dense_z_mu(x)
        if not alpha:
            return z_mu
        z_std = self.dense_z_std(x)
        return cat((z_mu, z_std), dim=1)


class Encoder2(NeuralNetwork):
    """ inception convolutional network """
    def __init__(self):
        super(Encoder2, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size)

        # Inception 1
        self.inc1_1 = Sequential(
            Conv2d(colors_dim, 4, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc1_2 = Sequential(
            Conv2d(colors_dim, 4, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv1 = Sequential(
            Conv2d(8, 8, kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU(),
        )

        # Inception 2
        self.inc2_1 = Sequential(
            Conv2d(8, 8, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc2_2 = Sequential(
            Conv2d(8, 8, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv2 = Sequential(
            Conv2d(16, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(16),
            ReLU(),
        )

        # Inception 3
        self.inc3_1 = Sequential(
            Conv2d(16, 16, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc3_2 = Sequential(
            Conv2d(16, 16, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv3 = Sequential(
            Conv2d(32, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU(),
        )

        # Inception 4
        self.inc4_1 = Sequential(
            Conv2d(32, 32, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc4_2 = Sequential(
            Conv2d(32, 32, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv4 = Sequential(
            Conv2d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            ReLU(),
        )

        self.dense1 = Sequential(
            Flatten(),
            Linear(1024, 256),
            ReLU(),
        )

        ## the following take the same input from dense1
        self.dense_z_mu = Linear(256, latent_dim)
        if alpha:
            self.dense_z_std = Sequential(
                Linear(256, latent_dim),
                Softplus(),
            )
        self.set_optimizer(optimizer, lr=learning_rate, betas=betas)

    def forward(self, x):
        inc1 = self.inc1_1(x)
        inc2 = self.inc1_2(x)
        x = self.conv1(cat((inc1,inc2), dim=1))
        inc1 = self.inc2_1(x)
        inc2 = self.inc2_2(x)
        x = self.conv2(cat((inc1,inc2), dim=1))
        inc1 = self.inc3_1(x)
        inc2 = self.inc3_2(x)
        x = self.conv3(cat((inc1,inc2), dim=1))
        inc1 = self.inc4_1(x)
        inc2 = self.inc4_2(x)
        x = self.conv4(cat((inc1,inc2), dim=1))
        x = self.dense1(x)
        z_mu = self.dense_z_mu(x)
        if not alpha:
            return z_mu
        z_std = self.dense_z_std(x)
        return cat((z_mu, z_std), dim=1)


class Encoder3(NeuralNetwork):
    """ convolutional network with leaky ReLU """
    def __init__(self):
        super(Encoder3, self).__init__()
        kernel_size = 5
        stride = 2
        padding = self.same_padding(kernel_size)

        self.conv0 = Sequential(
            Conv2d(colors_dim, 32, kernel_size=1, stride=1),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.conv1 = Sequential(
            Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(64, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.conv2 = Sequential(
            Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(128, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.dense1 = Sequential(
            Flatten(),
            Linear(32768, 1024),
            BatchNorm1d(1024, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )

        ## the following take the same input from dense1
        self.dense_z_mu = Linear(1024, latent_dim)
        if alpha:
            self.dense_z_std = Sequential(
                Linear(1024, latent_dim),
                Softplus(),
            )
        self.set_optimizer(optimizer, lr=learning_rate, betas=betas)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense1(x)
        z_mu = self.dense_z_mu(x)
        if not alpha:
            return z_mu
        z_std = self.dense_z_std(x)
        return cat((z_mu, z_std), dim=1)


class Encoder4(NeuralNetwork):
    """ convolutional network with BatchNorm and LeakyReLU """
    def __init__(self):
        super(Encoder4, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size)

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
            Linear(8192+labels_dim, 1024),
            BatchNorm1d(1024, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope)
        )

        ## the following take the same input from dense1
        self.dense_z_mu = Linear(1024, latent_dim)
        if alpha:
            self.dense_z_std = Sequential(
                Linear(1024, latent_dim),
                Softplus(),
            )
        self.set_optimizer(optimizer, lr=learning_rate, betas=betas)

    def forward(self, images, labels):
        x = self.conv0(images)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = cat((x, labels), dim=1)
        x = self.dense1(x)
        z_mu = self.dense_z_mu(x)
        if not alpha:
            return z_mu
        z_std = self.dense_z_std(x)
        return z_mu, z_std

