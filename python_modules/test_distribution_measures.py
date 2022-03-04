from functools import partial

import big.BigGAN2  as BigGAN
import dataset
from distribution_measures.evaluate_latent_distribution import evaluate_latent_distribution

batch_size = 8
num_workers = 4
labels_dim = 37

make_data_loader = dataset.MakeDataLoader(augmented=False)
data_loader_test = make_data_loader.get_data_loader_test(batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=num_workers)
data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers)
generator_biggan = partial(BigGAN.Generator, dim_z=128, labels_dim=labels_dim)
models = {"BigGAN": generator_biggan}
if __name__ == "__main__":
    print(evaluate_latent_distribution(models, data_loader_test, data_loader_valid))