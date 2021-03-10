""" pipelines of neural networks
"""

from neuralnetwork import NeuralNetwork
from encoder import Encoder4
from decoder import Decoder4
from discriminator import Discriminator4
from sampler import gaussian_sampler
from parameter import parameter

## all pipelines use the same networks
encoder = Encoder4().cuda()
decoder = Decoder4().cuda()
discriminator = Discriminator4().cuda()


class VAE(NeuralNetwork):
    """ conditional variational auto encoder """
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = gaussian_sampler

    def forward(self, images, labels):
        latent = self.encoder(images, labels)
        if parameter.alpha: ## in VAE mode, replace z by random sample
            latent = self.sampler(*latent)
        x_hat = self.decoder(latent,labels)
        return x_hat



class VAEGAN(NeuralNetwork):
    """ generative adversarial network with variationel autoencoder as generator  """
    def __init__(self):
        super(VAEGAN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.sampler = gaussian_sampler

    def forward(self, images, labels):
        latent = self.encoder(images, labels)
        if parameter.alpha:
            latent = self.sampler(*latent)
        generated_images = self.decoder(latent,labels)

        output = self.discriminator(generated_images)
        return output # true/fake, labels, metric
