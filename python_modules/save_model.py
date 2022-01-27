"""
Save complete models to file
"""
import torch

from file_system import folder_results
from big.BigGAN2 import Generator, Discriminator

def save_network(network, file: str):
    model = network()
    model.load()
    torch.save(model, file)

def save_generator():
    save_network(Generator, folder_results + "model_Generator.pth")

if __name__ == "__main__":
    save_generator()