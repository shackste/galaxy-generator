""" folder structure on google drive mounted to '/drive'
"""

import os

from google.colab import drive


root = "/drive/MyDrive/FHNW/galaxy_generator/"

## download from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
file_galaxy_images = root + "galaxyzoo_data_cropped_nonnormalized.npy"
file_galaxy_labels = root + "training_solutions_rev1.csv"

folder_results = root + "results/"

print(
    f"{file_galaxy_images.split('/')[0]}" +
    f" and {file_galaxy_labels.split('/')[0]}" +
    f" must be placed in google drive under " +
    f"{root.split('/')[2:].join('/')}"
)
print("the results will be placed there, too."

drive.mount("/drive")

if not os.path.exists(folder_results):
    os.mkdir(folder_results)
