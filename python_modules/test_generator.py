"""
This is a short skript to test a generator of galaxy images for mode collapse
"""
import torch

from sampler import generate_latent
from dataset import MakeDataLoader, DataLoader
from helpful_functions import write_generated_galaxy_images_iteration
from big.BigGAN2 import Generator
from conditional_autoencoder import ConditionalDecoder as Decoder

N_sample = 64


def test_model(model):
    # load labels
    make_data_loader = MakeDataLoader(N_sample=64, augmented=False)
    data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=N_sample, shuffle=False)
    for images_truth, labels in data_loader_valid:
        break
    # generate latent
    latent = generate_latent(labels.shape[0], model.dim_z, sigma=False, device="cpu")
    # create images
    images = model(latent, labels)
    # plot images
    write_generated_galaxy_images_iteration(iteration=0, images=images_truth.detach().cpu(), width=8, height=8, file_prefix="original_sample")
    write_generated_galaxy_images_iteration(iteration=1, images=images.detach().cpu(), width=8, height=8, file_prefix=f"test_generator_{type(model).__name__}_sample")


if __name__ == "__main__":
    for Gen in (Generator, Decoder):
        model = Gen()
        model.load()
        test_model(model)
