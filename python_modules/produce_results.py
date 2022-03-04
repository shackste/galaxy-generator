import torch
from pprint import pprint

from file_system import folder_results
from helpful_functions import write_generated_galaxy_images_iteration
from dataset import MakeDataLoader
from distribution_measures.evaluate_latent_distribution import evaluate_latent_distribution
from morphology_measures import measures
from morphology_measures.statistics import evaluate_generator, get_measures_dataloader, get_measures_generator, evaluate_measures

from conditional_autoencoder import ConditionalDecoder
from big.BigGAN2 import Generator

## results to be produced

plot_sample = False
compute_morphology = True
compute_cluster_measures = False


## load remaining models

models = {
    "BigGAN": Generator,
#    "cVAE" : ConditionalDecoder,
#   "info-scc" : InfoSCCGenerator,
#   "collapse" : CollapsedGenerator,
}

clean_morphology = {
    "deviation": (0, 999999),
    "asymmetry": (-1,1),
    "smoothness": (-0.5,1)

}

batch_size = 16

make_data_loader = MakeDataLoader()
data_loader_test = make_data_loader.get_data_loader_test(batch_size=batch_size, shuffle=True)
data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size, shuffle=True)

for images, labels in data_loader_test:
    break

### plot original images and reproduction from every model using labels and random latent

if compute_morphology:
    target_measures = get_measures_dataloader(data_loader_test)
    target_measures.clean_measures(clean_morphology)
    if True:
        baseline_measures = get_measures_dataloader(data_loader_test)
        baseline_measures.clean_measures(clean_morphology)
        distances = evaluate_measures(target_measures, baseline_measures, plot=True, name="baseline", plot_path=folder_results)
        pprint(distances)


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
            generated_measures = get_measures_generator(model, data_loader_valid)
            generated_measures.clean_measures(clean_morphology)
            distances = evaluate_measures(target_measures, generated_measures, plot=True, name=name, plot_path=folder_results)
            pprint(distances)
#            distances, measures_target = evaluate_generator(data_loader_test, model, name=name, plot=True, test=True, plot_path=folder_results, clean=clean_morphology)
#            pprint(distances)


### compute values to be put into the table

## measures on 16 features reduction
if compute_cluster_measures:
    pprint(evaluate_latent_distribution(models, data_loader_test, data_loader_valid, N_cluster=10))
# in the first dict, we want the errors and distances, the second dict contains the wasserstein distances


## add Vitaliys measures

