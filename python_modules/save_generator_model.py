import torch

from big.BigGAN2 import Generator
from file_system import folder_results


labels_dim = 37
latent_dim = 128
lr_generator = 5e-5  # 1e-3 #5es-5

generator = Generator(dim_z=latent_dim, labels_dim=labels_dim, G_lr=lr_generator)
generator.load()

filename = folder_results + "model_Generator.pth"

torch.save(generator, filename)

## load model
## generator = torch.load(filename)
