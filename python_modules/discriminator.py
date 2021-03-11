""" Discriminator networks for RGB images """

from torch import cat
from torch.nn import Sequential, \
                     Conv2d, Linear, \
                     ReLU, LeakyReLU, Sigmoid, Softmax, \
                     BatchNorm1d, BatchNorm2d, Flatten


from neuralnetwork import NeuralNetwork
from parameter import colors_dim, labels_dim, parameter

class Discriminator1(NeuralNetwork):
    def __init__(self):
        super(Discriminator1, self).__init__()
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
            Linear(1024,256),
            ReLU(),
        )
        self.dense2 = Sequential(
            Linear(256,1),
            Sigmoid(),
        )
        self.set_optimizer(parameter.optimizer, lr=parameter.learning_rate, betas=parameter.betas)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class Discriminator3(NeuralNetwork):
    def __init__(self):
        super(Discriminator3, self).__init__()
        kernel_size = 5
        stride = 2
        padding = self.same_padding(kernel_size)

        self.conv0 = Sequential(
            Conv2d(colors_dim, 32, kernel_size=1, stride=1),
            LeakyReLU(negative_slope=parameter.negative_slope),
        )
        self.conv1 = Sequential(
            Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(64, momentum=parameter.momentum),
            LeakyReLU(negative_slope=parameter.negative_slope),
        )
        self.conv2 = Sequential(
            Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(128, momentum=parameter.momentum),
            LeakyReLU(negative_slope=parameter.negative_slope),
        )
        self.dense1 = Sequential(
            Flatten(),
            Linear(32768,1024),
            BatchNorm1d(1024, momentum=parameter.momentum),
            LeakyReLU(negative_slope=parameter.negative_slope),
        )
        self.dense2 = Sequential(
            Linear(1024,1),
            Sigmoid(),
        )
        self.set_optimizer(parameter.optimizer, lr=parameter.learning_rate, betas=parameter.betas)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class Discriminator4(NeuralNetwork):
    def __init__(self):
        super(Discriminator4, self).__init__()
        kernel_size = 3
        stride = 2
        padding = self.same_padding(kernel_size)
        self.conv0 = Sequential(
            Conv2d(colors_dim, 16, kernel_size=1, stride=1),
            LeakyReLU(negative_slope=parameter.negative_slope),
        )
        self.conv1 = Sequential(
            Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(32, momentum=parameter.momentum),
            LeakyReLU(negative_slope=parameter.negative_slope)
        )
        self.conv2 = Sequential(
            Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(64, momentum=parameter.momentum),
            LeakyReLU(negative_slope=parameter.negative_slope)
        )
        self.conv3 = Sequential(
            Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(128, momentum=parameter.momentum),
            LeakyReLU(negative_slope=parameter.negative_slope)
        )
        self.dense1 = Sequential(
            Flatten(),
            Linear(8192, 1024),
            LeakyReLU(negative_slope=parameter.negative_slope)
        )   ## outputs metric

        ## the following take metric as input
        self.dense2_1 = Sequential(
            Linear(1024,1),
            Sigmoid(),
        )   ## outputs true/fake
        self.dense2_2 = Sequential(
            Linear(1024, labels_dim),
            Softmax(dim=1),
        )   ## outputs labels

        self.set_optimizer(parameter.optimizer, lr=parameter.learning_rate, betas=parameter.betas)

    def forward(self, images):
        x = self.conv0(images)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        metric = self.dense1(x)
        true_fake = self.dense2_1(metric)
        labels = self.dense2_2(metric)
        return cat((true_fake, labels, metric), dim=1)

