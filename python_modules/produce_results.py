import torch
from pprint import pprint

from file_system import folder_results
from helpful_functions import write_generated_galaxy_images_iteration
from dataset import MakeDataLoader
from distribution_measures.evaluate_latent_distribution import evaluate_latent_distribution
from morphology_measures.statistics import evaluate_generator

from conditional_autoencoder import ConditionalDecoder
from big.BigGAN2 import Generator

## results to be produced

plot_sample = False
compute_morphology = True
compute_cluster_measures = False


## load remaining models

models = {
    "BigGAN": Generator,
    "cVAE" : ConditionalDecoder,
#   "info-scc" : InfoSCCGenerator,
#   "collapse" : CollapsedGenerator,
}

batch_size = 16

make_data_loader = MakeDataLoader()
data_loader_test = make_data_loader.get_data_loader_test(batch_size=batch_size, shuffle=True)
data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size, shuffle=True)

for images, labels in data_loader_test:
    break

### plot original images and reproduction from every model using labels and random latent

if plot_sample or compute_morphology:
    if plot_sample:
        write_generated_galaxy_images_iteration(iteration=0, images=images, width=4, height=4, file_prefix="images_original")
    for name, Model in models.items():
        model = Model().cuda()
        model.load()
        model.eval()
        print(name)
        with torch.no_grad():
            latent = torch.randn(batch_size, model.dim_z, device="cuda")
            labels = labels.cuda()
            generated_images = model(latent, labels)
        write_generated_galaxy_images_iteration(iteration=0, images=generated_images, width=4, height=4,
                                                file_prefix=f"images_{name}")

        ## morphology measures, takes a while. For testing, use test=True
        if compute_morphology:
            pprint(evaluate_generator(data_loader_test, model, name=name, plot=True, test=True, plot_path=folder_results))


### compute values to be put into the table

## measures on 16 features reduction
if compute_cluster_measures:
    pprint(evaluate_latent_distribution(models, data_loader_test, data_loader_valid, N_cluster=10))
# in the first dict, we want the errors and distances, the second dict contains the wasserstein distances


## add Vitaliys measures

