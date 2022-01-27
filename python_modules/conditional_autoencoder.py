"""
This file contains the basic conditional Variational Autoencoder (cVAE) used to reproduce and generate galaxy images.
The classifier used for training is the same as used for training of BigGAN.
"""
from tqdm import trange

from torch import cat
from torch.optim import Adam
from torch.nn import Sequential, ModuleList, \
                     Conv2d, Linear, \
                     LeakyReLU, Softplus, Tanh, \
                     BatchNorm1d, BatchNorm2d, Flatten, \
                     ConvTranspose2d, UpsamplingBilinear2d

from sampler import gaussian_sampler
from loss import loss_reconstruction,loss_kl, loss_class
from dataset import MakeDataLoader
from neuralnetwork import NeuralNetwork
from image_classifier import ImageClassifier

# parameters for cVAE
colors_dim = 3
labels_dim = 37
momentum = 0.99 # Batchnorm
negative_slope = 0.2 # LeakyReLU
latent_dim = 128
optimizer = Adam
learning_rate = 2e-4
betas = (0.5, 0.999)


def train_autoencoder(epochs: int = 250, batch_size: int = 64, alpha: float = 0.0005, beta: float = 0.5, num_workers=4):
    """ perform training loop for the cVAE """
    encoder = ConditionalEncoder().cuda()
    decoder = ConditionalDecoder().cuda()
    classifier = ImageClassifier().cuda()
    classifier.load()
    make_data_loader = MakeDataLoader(augmented=True)
    data_loader = make_data_loader.get_data_loader_train(batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=num_workers)
    for epoch in trange(epochs, desc="epochs"):
        for images, labels in data_loader:
            images, labels = images.cuda(), labels.cuda()
            images = images*2 - 1 # rescale (0,1) to (-1,1)

            latent_mu, latent_sigma = encoder(images, labels)
            latent = gaussian_sampler(latent_mu, latent_sigma)
            generated_images = decoder(latent, labels)
            generated_labels = classifier(generated_images)

            decoder.zero_grad()
            encoder.zero_grad()
            lr = loss_reconstruction(images, generated_images)
            g_loss = lr + alpha*loss_kl(latent)+ beta*loss_class(labels, generated_labels)
            g_loss.backward()

            encoder.optimizer.step()
            decoder.optimizer.step()
        decoder.save()


class ConditionalEncoder(NeuralNetwork):
    """ convolutional network with BatchNorm and LeakyReLU """
    def __init__(self, dim_z=latent_dim):
        super(ConditionalEncoder, self).__init__()
        self.dim_z = dim_z
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
            Linear(8192, 2048),
            BatchNorm1d(2048, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope)
        )
        self.dense2 = Sequential(
            Linear(2048, self.dim_z),
            BatchNorm1d(self.dim_z, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope)
        )
        self.embedding = Sequential(
            Linear(labels_dim, self.dim_z),
            BatchNorm1d(self.dim_z, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )


        ## the following take the same input from dense1
        self.dense_z_mu = Linear(128*2, self.dim_z)
        self.dense_z_std = Sequential(
            Linear(self.dim_z*2, self.dim_z),
            Softplus(),)
        self.set_optimizer(optimizer, lr=learning_rate, betas=betas)

    def forward(self, images, labels):
        x = self.conv0(images)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.embedding(labels)
        x = cat((x, y), dim=1)
        z_mu = self.dense_z_mu(x)
        z_std = self.dense_z_std(x)
        return z_mu, z_std



class ConditionalDecoder(NeuralNetwork):
    def __init__(self, ll_scaling=1.0, dim_z=latent_dim):
        super(ConditionalDecoder, self).__init__()
        self.dim_z = dim_z
        ngf = 32
        self.init = genUpsample(self.dim_z, ngf * 16, 1, 0)
        self.embedding = Sequential(
            Linear(labels_dim, self.dim_z),
            BatchNorm1d(self.dim_z, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.dense_init = Sequential(
            Linear(self.dim_z*2, self.dim_z),
            BatchNorm1d(self.dim_z, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.m_modules = ModuleList()#to 4x4
        self.c_modules  = ModuleList()
        for i in range(4):
          self.m_modules.append(genUpsample2(ngf * 2**(4-i), ngf * 2**(3-i),3))
          self.c_modules.append(Sequential(Conv2d(ngf * 2**(3-i), colors_dim, 3, 1, 1, bias=False), Tanh()))
        self.set_optimizer(optimizer, lr=learning_rate*ll_scaling, betas=betas)

    def forward(self, latent, labels, step=3, alpha=1):
        y = self.embedding(labels)
        out = cat((latent, y), dim=1)
        out = self.dense_init(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = self.init(out)
        for i in range(step):
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
