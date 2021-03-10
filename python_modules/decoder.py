""" Decoder networks for RGB images """

from torch import cat
from torch.nn import Sequential, \
                     Conv2d, ConvTranspose2d, Linear, \
                     ReLU, LeakyReLU, Sigmoid, \
                     BatchNorm1d, BatchNorm2d, Flatten


from neuralnetwork import NeuralNetwork
from additional_layers import Reshape
from parameter import colors_dim, labels_dim, parameter

# intermediate tensor shape flat -> original, derived from encoder
flat_dim = 8192
orig_dim = (128, 8, 8)


class Decoder1(NeuralNetwork):
    """ reverse basic convolutional network """
    def __init__(self):
        super(Decoder1, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size) 
        self.dense1 = Sequential(
            Linear(parameter.latent_dim, flat_dim),
            BatchNorm1d(flat_dim),
            ReLU(),
            Reshape(*orig_dim)
        )
        self.conv1 = Sequential(
            ConvTranspose2d(orig_dim[0], 32, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            ReLU(),
        )
        self.conv2 = Sequential(
            ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            ReLU(),
        )
        self.conv3 = Sequential(
            ConvTranspose2d(16, 8, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            BatchNorm2d(8),
            ReLU(),
        )
        self.conv4 = Sequential(
            ConvTranspose2d(8, colors_dim, kernel_size=1, stride=1),
            Sigmoid(),
        )
        self.set_optimizer(parameter.optimizer, lr=parameter.learning_rate, betas=parameter.betas)

    def forward(self, x):
        x = self.dense1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Decoder2(NeuralNetwork):
    """ reverse inception convolutional network """
    def __init__(self):
        super(Decoder2, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size) 
        self.dense1 = Sequential(
            Linear(parameter.latent_dim, flat_dim),
            BatchNorm1d(flat_dim),
            ReLU(),
            Reshape(*orig_dim)
        )
        # inception 1
        self.inc1_1 = Sequential(
            Conv2d(orig_dim[0], 16, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc1_2 = Sequential(
            Conv2d(orig_dim[0], 16, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv1 = Sequential(
            ConvTranspose2d(32, 32, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            ReLU(),
        )
        # inception 2
        self.inc2_1 = Sequential(
            Conv2d(32, 8, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc2_2 = Sequential(
            Conv2d(32, 8, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv2 = Sequential(
            ConvTranspose2d(16, 16, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            ReLU(),
        )
        # inception 3
        self.inc3_1 = Sequential(
            Conv2d(16, 4, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc3_2 = Sequential(
            Conv2d(16, 4, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv3 = Sequential(
            ConvTranspose2d(8, 8, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            BatchNorm2d(8),
            ReLU(),
        )
        # inception 4
        self.inc4_1 = Sequential(
            Conv2d(8, 2, kernel_size=1, stride=1),
            ReLU(),
        )
        self.inc4_2 = Sequential(
            Conv2d(8, 2, kernel_size=kernel_size, stride=1, padding=padding),
            ReLU(),
        )
        self.conv4 = Sequential(
            ConvTranspose2d(4, 4, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            Sigmoid(),
        )
        self.set_optimizer(parameter.optimizer, lr=parameter.learning_rate, betas=parameter.betas)

    def forward(self, x):
        x = self.dense1(x)
        x1 = self.inc1_1(x)
        x2 = self.inc1_2(x)
        x = self.conv1(cat((x1, x2), dim=1))
        x1 = self.inc2_1(x)
        x2 = self.inc2_2(x)
        x = self.conv2(cat((x1, x2), dim=1))
        x1 = self.inc3_1(x)
        x2 = self.inc3_2(x)
        x = self.conv3(cat((x1, x2), dim=1))
        x1 = self.inc4_1(x)
        x2 = self.inc4_2(x)
        x = self.conv4(cat((x1, x2), dim=1))
        return x


class Decoder3(NeuralNetwork):
    """ reverse convolutional network with kernel of 5"""
    def __init__(self):
        super(Decoder3, self).__init__()
        kernel_size = 5
        stride = 2
        padding = self.same_padding(kernel_size) 
        self.dense1 = Sequential(
            Linear(parameter.latent_dim, 1024),
            BatchNorm1d(1024, momentum=parameter.momentum),
            ReLU(),
        )
        self.dense2= Sequential(
            Linear(1024, flat_dim),
            BatchNorm1d(8*8*128, momentum=parameter.momentum),
            ReLU(),
            Reshape(*orig_dim)
        )
        self.conv1 = Sequential(
            ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            BatchNorm2d(64, momentum=parameter.momentum),
            ReLU(),
        )
        self.conv2 = Sequential(
            ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            BatchNorm2d(32, momentum=parameter.momentum),
            ReLU(),
        )
        self.conv3 = Sequential(
            ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
            BatchNorm2d(16, momentum=parameter.momentum),
            ReLU(),
        )
        self.conv4 = Sequential(
            ConvTranspose2d(16, colors_dim, kernel_size=1),
            Sigmoid()
        )
        self.set_optimizer(parameter.optimizer, lr=parameter.learning_rate, betas=parameter.betas)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Decoder4(NeuralNetwork):
    """ reverse convolutional network with kernel of 5"""
    def __init__(self):
        super(Decoder4, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size) 
        self.dense1 = Sequential(
            Linear(parameter.latent_dim + labels_dim, 1024),
            BatchNorm1d(1024, momentum=parameter.momentum),
            ReLU(),
        )
        self.dense2 = Sequential(
            Linear(1024, flat_dim),
            BatchNorm1d(flat_dim, momentum=parameter.momentum),
            ReLU(),
            Reshape(*orig_dim),
        )
        self.conv1 = Sequential(
            ConvTranspose2d(orig_dim[0], 64, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            BatchNorm2d(64, momentum=parameter.momentum),
            ReLU(),
        )
        self.conv2 = Sequential(
            ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            BatchNorm2d(32, momentum=parameter.momentum),
            ReLU(),
        )
        self.conv3 = Sequential(
            ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding),
            BatchNorm2d(16, momentum=parameter.momentum),
            ReLU(),
        )
        self.conv4 = Sequential(
            ConvTranspose2d(16, colors_dim, kernel_size=1, stride=1),
            Sigmoid(),
        )
        self.set_optimizer(parameter.optimizer, lr=parameter.learning_rate, betas=parameter.betas)

    def forward(self, latent, labels):
        x = self.dense1(cat((latent, labels), dim=1))
        x = self.dense2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
        
