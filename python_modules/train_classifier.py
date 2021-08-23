import numpy as np
import torch
from torch.optim import SGD, Adam
from tqdm import trange
from functools import partial

from image_classifier import ImageClassifier as Classifier
from dataset import MakeDataLoader

epochs = 2
reload = False ## if True, continue with previously trained parameters

considered_layers_init = [1, 2, 3, 4, 5][:4]

# setting used by Dielemann et al 2015
#optimizer = SGD
#optimizer_kwargs = {"nesterov":True, "momentum":0.9}
optimizer = Adam
optimizer_kwargs = {}
learning_rate_init = 0.04
gamma = 0.995 # learning rate decay factor
sample_variance_threshold = 0.002
seed_parameter = 7
weight_loss_sample_variance = 0 #200.

batch_size = 16
N_batches = 1
N_sample = -1 #batch_size * N_batches
evaluation_steps = 250 # N_batches*10

track = True


hyperparameter_dict = {
    "lr_init" : learning_rate_init,
    "lr_gamma" : gamma,
    "seed_parameter" : seed_parameter,
}

wandb_kwargs = {
    "project" : "galaxy classifier", ## top level identifier
    "group" : "parameter search", ## secondary identifier
    "job_type" : "training", ## third level identifier
    "tags" : ["training", "parameter search"],  ## tags for organizing tasks
    "name" : "test", ## bottom level identifier, label of graph in UI
    "config" : hyperparameter_dict, ## dictionary of used hyperparameters
}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    N_gpus = torch.cuda.device_count()
    batch_size *= N_gpus
else:
    N_gpus = None


def train_classifier(classifier: Classifier, make_data_loader, *, epochs: int = 5, batch_size: int = 32, save: bool = False, track: bool = False):
    schedule = {
        # epoch : performed change
        2 : classifier.use_label_hierarchy,
    }

    for epoch in trange(epochs, desc=f"epochs"):
        data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size) #, num_workers=4)
        data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size) #, num_workers=4)

        if classifier.epoch in schedule.keys():
            schedule[classifier.epoch]()
        classifier.train_epoch(data_loader_train, data_loader_valid, track=track)
        classifier.plot_losses(save=save)
        classifier.plot_accuracy(save=save)
        classifier.plot_test_accuracy(save=save)
        classifier.plot_sample_variances(save=save)

def train_classifier_tracked(*args, wandb_kwargs: dict = wandb_kwargs, **kwargs):
    from track_progress import track_progress
    train = partial(train_classifier, *args, **kwargs)
    track_progress(train, wandb_kwargs=wandb_kwargs)


def train_classifier_on_hyperparameters(learning_rate_init=learning_rate_init, gamma=gamma, seed_parameter=seed_parameter, track=track):
    hyperparameter_dict = {
        "lr_init": learning_rate_init,
        "lr_gamma": gamma,
        "seed_parameter": seed_parameter,
    }
    wandb_kwargs.update({"config":hyperparameter_dict})

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

    if N_gpus > 1  and device.type == "cuda":
        classifier = torch.nn.DataParallel(classifier)

    if reload:
        classifier.load()
        classifier.use_label_hierarchy()
        
#    with torch.autograd.detect_anomaly():
    if track:
        train_classifier_tracked(classifier, make_data_loader, epochs=epochs, save=True, batch_size=batch_size, wandb_kwargs=wandb_kwargs, track=True)
    else:
        train_classifier(classifier, make_data_loader, epochs=epochs, save=True, batch_size=batch_size)

def train_classifier_on_random_hyperparameters(learning_rate_init=None, gamma=None, seed_parameter=None, track=track):
    if not learning_rate_init:
        learning_rate_init = 10.**(-2 + 3*np.random.random())
    if not gamma:
        gamma = np.random.lognormal(-0.005, 0.0015)
    if not seed_parameter:
        seed_parameter = np.random.randint(200)
    print(seed_parameter, learning_rate_init, gamma)
    train_classifier_on_hyperparameters(learning_rate_init=learning_rate_init, gamma=gamma, seed_parameter=seed_parameter, track=track)


if __name__ == "__main__":

    train_classifier_on_random_hyperparameters(seed_parameter=seed_parameter)