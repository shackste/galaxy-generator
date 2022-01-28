import torch

from helpful_functions import write_generated_galaxy_images_iteration
from dataset import MakeDataLoader
from distribution_measures.evaluate_latent_distribution import evaluate_latent_distribution

from conditional_autoencoder import ConditionalDecoder
from big.BigGAN2 import Generator

## load remaining models

models = {
    "BigGAN" : Generator,
    "cVAE" : ConditionalDecoder,
#   "info-scc" : InfoSCCGenerator,
#   "collapse" : CollapsedGenerator,
}

batch_size = 16

make_data_loader = MakeDataLoader()
data_loader_test = make_data_loader.get_data_loader_test(batch_size=batch_size, shuffle=True)
data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size, shuffle=True)

for images, label in data_loader_test:
    break

### plot original images and reproduction from every model using labels and random latent

write_generated_galaxy_images_iteration(iteration=0, images=images, width=4, height=4, file_prefix="images_original")
for name, Model in models.items():
    model = Model()
    model.load()
    model.eval()
    latent = torch.randn(batch_size, model.z_dim)
    generated_images = model(latent, labels)
    write_generated_galaxy_images_iteration(iteration=0, images=generated_images, width=4, height=4,
                                            file_prefix=f"images_{name}")


### compute values to be put into the table

print(evaluate_latent_distribution(models, data_loader_test, data_loader_valid, N_cluster=10))
# in the first dict, we want the errors and distances, the second dict contains the wasserstein distances


## add Vitaliys measures

## add morphology measures
### also including corner plots