import matplotlib.pyplot as plt


from image_classifier import ImageClassifier
from dataset import MakeDataLoader
from helpful_functions import write_RGB_image
from file_system import folder_results

## load augmentation procedure
cls = ImageClassifier()
augment = cls.augment
## load single image
make_data_loader = MakeDataLoader()
data_loader = make_data_loader.get_data_loader_train(batch_size=1,)
for image, _ in data_loader:
    break
## produce augmentations
augmentations = augment(image)

## plot single image and augmented images
write_RGB_image(image=image[0].numpy(), filename="original_image.png")
for i, a in enumerate(augmentations):
    write_RGB_image(image=a.numpy(), filename=f"augmentation{i}.png")
