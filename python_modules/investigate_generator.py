import torch

from helpful_functions import write_generated_galaxy_images_iteration
from labeling import make_galaxy_labels_hierarchical
from big.BigGAN2 import Generator


latent_dim = 128
labels_dim = 37

galaxy_type_label = {                     # Q1    Q2    Q3   Q4   Q5       Q6   Q7     Q8             Q9     Q10    Q11
    "spiral, 1-arm" :         torch.tensor([0,1,0, 0,1, 0,1, 1,0, 1,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 1,0,0,0,0,0]).float(),
    "spiral, 2-arms" :        torch.tensor([0,1,0, 0,1, 0,1, 1,0, 1,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,1,0,0,0,0]).float(),
    "spiral, 3-arms" :        torch.tensor([0,1,0, 0,1, 0,1, 1,0, 1,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,0,1,0,0,0]).float(),
    "spiral, 4-arms" :        torch.tensor([0,1,0, 0,1, 0,1, 1,0, 1,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,0,0,1,0,0]).float(),
    "spiral, 2-arms, tight" : torch.tensor([0,1,0, 0,1, 0,1, 1,0, 1,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,1,0,0,0,0]).float(),
    "spiral, 2-arms, loose" : torch.tensor([0,1,0, 0,1, 0,1, 1,0, 1,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,1,0,0,0,0]).float(),
    "spiral, small bulge" :   torch.tensor([0,1,0, 0,1, 0,1, 1,0, 0,1,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,1,0,0,0,0]).float(),
    "spiral, med bulge" :     torch.tensor([0,1,0, 0,1, 0,1, 1,0, 0,0,1,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,1,0,0,0,0]).float(),
    "spiral, big bulge" :     torch.tensor([0,1,0, 0,1, 0,1, 1,0, 0,0,0,1, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,1,0,0,0,0]).float(),
    "spiral, bar" :           torch.tensor([0,1,0, 0,1, 1,0, 1,0, 1,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,1,0, 0,1,0,0,0,0]).float(),
    "disk, edge-on" :         torch.tensor([0,1,0, 1,0, 0,0, 0,0, 0,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,0,1, 0,0,0, 0,0,0,0,0,0]).float(),
    "disk, edge-on, boxy" :   torch.tensor([0,1,0, 1,0, 0,0, 0,0, 0,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 0,1,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "disk, edge-on, round" :  torch.tensor([0,1,0, 1,0, 0,0, 0,0, 0,0,0,0, 0,1, 0,0,0, 0,0,0,0,0,0,0, 1,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "elliptical, cigar" :     torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 0,1, 0,0,1, 0,0,0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "elliptical" :            torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 0,1, 0,1,0, 0,0,0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "elliptical, blob":       torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 0,1, 1,0,0, 0,0,0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "ring" :                  torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 1,0, 1,0,0, 1,0,0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "arc":                    torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 1,0, 1,0,0, 0,1,0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "disturbed" :             torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 1,0, 1,0,0, 0,0,1,0,0,0,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "irregular" :             torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 1,0, 1,0,0, 0,0,0,1,0,0,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "merger" :                torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 1,0, 1,0,0, 0,0,0,0,0,1,0, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
    "dust lane":              torch.tensor([1,0,0, 0,0, 0,0, 0,0, 0,0,0,0, 1,0, 1,0,0, 0,0,0,0,0,0,1, 0,0,0, 0,0,0, 0,0,0,0,0,0]).float(),
}

generator = Generator()
generator.load()


@torch.no_grad()
def plot_generated_galaxy_type(galaxy_type: str, N: int = 64, noise: float = 0.2):
    """ plot N images of a certain galaxy type """
    label = galaxy_type_label[galaxy_type]
    labels = label.repeat(N).reshape(N,labels_dim)
    if noise:
        labels += noise * torch.randn(N, labels_dim)
        labels[labels < 0] = 0
        labels = make_galaxy_labels_hierarchical(labels)
    latent = torch.randn(N, latent_dim)
    images = generator(latent, labels)
    width = min(8, N)
    write_generated_galaxy_images_iteration(iteration=0, images=images.cpu(), width=width, height=N//width, file_prefix="generate_" + galaxy_type)

def investigate_generator():
    for galaxy_type in galaxy_type_label.keys():
        plot_generated_galaxy_type(galaxy_type)


if __name__ == "__main__":
    investigate_generator()