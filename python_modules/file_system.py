""" folder structure on google drive mounted to '/drive'
"""

import os

#root = "/drive/MyDrive/FHNW/galaxy_generator/"
#root = "/home/shackste/galaxy-generator/data/"
root = "/mnt/data/shackste/galaxy-generator/data/"

## download from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
file_galaxy_images = root + "galaxyzoo_data_cropped_nonnormalized.npy"
file_galaxy_labels = root + "training_solutions_rev1.csv"
folder_images = root + "images_training_rev1/" #images_training_rev1/"

folder_results = root + "results/"

print("\n!!!!!!!!!!\n")
print(
    f"{file_galaxy_images.split('/')[-1]}" +
    f" and {file_galaxy_labels.split('/')[-1]}" +
    f" must be placed in google drive under " +
    f"{'/'.join(root.split('/')[3:])}"
)
print("the results will be placed there, too.\n")

if not os.path.exists(folder_results):
    os.mkdir(folder_results)
