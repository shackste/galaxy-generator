import torch

from image_classifier import ImageClassifier
from dataset import MakeDataLoader

classifier = ImageClassifier()
classifier.load()
classifier.eval()
classifier.use_label_hierarchy()
make_data_loader = MakeDataLoader()
data_loader = make_data_loader.get_data_loader_valid(batch_size=64,
                                                     shuffle=True,
                                                     num_workers=4)
for images, labels in data_loader:
    predicted_labels = classifier(images)
    mse = torch.mean((labels - predicted_labels)**2)
    print("rmse", torch.sqrt(mse))
    print()
    print("L1", torch.mean(torch.abs(labels - predicted_labels)))
    break