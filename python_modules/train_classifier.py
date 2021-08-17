import torch
from torch.optim import SGD
from tqdm import trange

from image_classifier import ImageClassifier as Classifier
from dataset import MakeDataLoader

epochs = 500
reload = False ## if True, continue with previously trained parameters

considered_layers_init = [1, 2, 3, 4, 5][:4]

optimizer = SGD
optimizer_kwargs = {"nesterov":True, "momentum":0.9}
learning_rate_init = 0.04
gamma = 0.995 # learning rate decay factor
gamma = 0.1
sample_variance_threshold = 0.002
seed_parameter = 7
weight_loss_sample_variance = 0 #200.

batch_size = 16
N_batches = 1
N_sample = -1 #batch_size * N_batches
evaluation_steps = 250 # N_batches*10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classifier(classifier: Classifier, make_data_loader, epochs=5, batch_size=32, save=False):
    schedule = {
        # epoch : performed change
        2 : classifier.use_label_hierarchy,
    }

    for epoch in trange(epochs, desc=f"epochs"):
        data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size) #, num_workers=4)
        data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size) #, num_workers=4)

        if classifier.epoch in schedule.keys():
            schedule[classifier.epoch]()
        classifier.train_epoch(data_loader_train, data_loader_valid)
        classifier.plot_losses(save=save)
        classifier.plot_accuracy(save=save)
        classifier.plot_test_accuracy(save=save)
        classifier.plot_sample_variances(save=save)

if __name__ == "__main__":
    make_data_loader = MakeDataLoader(N_sample=N_sample)
    classifier = Classifier(seed=seed_parameter,
                            gamma=gamma,
                            sample_variance_threshold=sample_variance_threshold,
                            optimizer=optimizer,
                            optimizer_kwargs=optimizer_kwargs, 
                            learning_rate_init=learning_rate_init,
                            weight_loss_sample_variance=weight_loss_sample_variance,
                            evaluation_steps=evaluation_steps,
                            considered_layers_init=considered_layers_init,
                           ).to(device)

    if reload:
        classifier.load()
        classifier.use_label_hierarchy()
        
#    with torch.autograd.detect_anomaly():
    train_classifier(classifier, make_data_loader, epochs=epochs, save=True, batch_size=batch_size)
