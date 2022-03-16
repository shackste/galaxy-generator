# python scripts

This folder contains all python scripts for models with the purpose of generating unseen galaxy images as found in the dataset provided with the galaxy zoo - the galaxy challenge competition on kaggle (https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).
These scripts include the models themselves, their training loops, handling of the dataset as well as a number of evaluation metrics.

### Models
conditional variational autoencoder

BigGAN + Classifier

### Metrics

based on morphological properties, subfolder morphology_measures/

based on k-means clusters, subfolder distribution_measures/

commonly used in ML (IS, FID, KID, PPL, Chamfer distance, geometric distance, attribute accuracy)
located in subfolder evaluation/