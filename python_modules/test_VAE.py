import torch

from distribution_measures import autoencoder
from helpful_functions import write_generated_galaxy_images_iteration as write


decoder = autoencoder.Decoder()

print(decoder.parameter_file)

decoder.load()

latent = torch.randn(64,8)

images = decoder(latent)

print(images.min(), images.max())

write(iteration=0, images=images, file_prefix="test_VAEreduc")
images = (images+1)/2
print(images.min(), images.max())
write(iteration=1, images=images, file_prefix="test_VAEreduc")