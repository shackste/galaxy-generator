# Evaluation of distribution of morphological properties

Contained in this folder is the code for evaluation of morhpological properties of galaxy images, namely ellipticity, gini-m20, MID and CAS.
These are computed using the statmorph code (https://statmorph.readthedocs.io/en/latest/).
The evaluation metric is obtained by first computing the properties for a subset taking from the training images.
Then, images are generated using labels from a separate set of identical size and properties are computied for the generated set.
Finally, the distance between these two point clouds is computed using the Wasserstein distance.

The code can be used for all kinds of conditional generators that take as input two pytorch tensors: latent and labels.
The generator should include a parameter z_dim, that indicates the dimension of required latent vectors.
The DataLoaders passed to the code should return a tuple of (images, labels).


## Usage


```
from statistics import *
## compute morphological properties for training images
target_measures = get_measures_dataloader(data_loader_test)
## clean values outside the intended distribution
target_measures.clean_measures(clean_morphology)
## compute morphological properties for generated images. model is a pre-trained instance of the generator
generated_measures = get_measures_generator(model, data_loader_valid)
## clean values outside the intended distribution
generated_measures.clean_measures(clean_morphology)
## compute Wasserstein distance between point clouds
distances = evaluate_measures(target_measures, generated_measures, plot=True, name=name, plot_path=folder_results)
```

The last function provides a dict that contains Wasserstein distances for each group of properties, as well as for all properties combined.

