""" folder structure on google drive mounted to '/drive' via

from google.colab import drive
drive.mount("/drive")
"""

import os

root = "/drive/MyDrive/FHNW/galaxy_generator/"

## the following files need to be downloaded to your google drive in root folder defined above
## download from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
file_galaxy_images = root + "galaxyzoo_data_cropped_nonnormalized.npy"
file_galaxy_labels = root + "training_solutions_rev1.csv"

folder_results = root + "results_pytorch/"

if not os.path.exists(folder_results):
    os.mkdir(folder_results)
