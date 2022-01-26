import numpy as np
import torch
from torch.optim import SGD, Adam
from tqdm import trange
from functools import partial

from image_classifier import ImageClassifier as Classifier
from dataset import MakeDataLoader

epochs = 3000
reload = False ## if True, continue with previously trained parameters

considered_groups = list(range(1,12))

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

batch_size = 64
N_batches = 1
N_sample = -1 #batch_size * N_batches
evaluation_steps = 1000 #865 #250 # N_batches*10
N_batches_test = 90000 # number of batches considered for evaluation
num_workers = 24

track = True


hyperparameter_dict = {
    "lr_init" : learning_rate_init,
    "lr_gamma" : gamma,
    "seed_parameter" : seed_parameter,
}

wandb_kwargs = {
    "project" : "galaxy classifier", ## top level identifier
    "group" : "distributed", ## secondary identifier
    "job_type" : "full training", ## third level identifier
    "tags" : ["training"],  ## tags for organizing tasks
    "name" : f"lr {learning_rate_init}, 8 GPUs", ## bottom level identifier, label of graph in UI
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
        1 : classifier.use_label_hierarchy,
    }
    classifier.use_label_hierarchy()

    data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size, num_workers=num_workers)
    data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size, num_workers=num_workers)
    for epoch in trange(epochs, desc=f"epochs"):
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
    wandb_kwargs["name"] = f"lr {learning_rate_init:.3f}, gamma{gamma:.4f}"


    make_data_loader = MakeDataLoader(N_sample=N_sample)
    classifier = Classifier(seed=seed_parameter,
                            gamma=gamma,
                            sample_variance_threshold=sample_variance_threshold,
                            optimizer=optimizer,
                            optimizer_kwargs=optimizer_kwargs, 
                            learning_rate_init=learning_rate_init,
                            weight_loss_sample_variance=weight_loss_sample_variance,
                            evaluation_steps=evaluation_steps,
                            considered_groups=considered_groups,
                            N_batches_test=N_batches_test,
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


### distributed on multiple GPUs
import os
from collections import Counter

import wandb
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR

from file_system import folder_results
from accuracy_measures import measure_accuracy_classifier
from loss import mse

def train_classifier_distributed(rank, world_size, optimizer=optimizer):
    # initialize distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(rank=rank, world_size=world_size, backend="nccl")
    # prepare splitted dataloader
    make_data_loader = MakeDataLoader()
    sampler_train = DistributedSampler(make_data_loader.dataset_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size, num_workers=0, sampler=sampler_train, shuffle=False)
    data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size, num_workers=0)
    # prepare model
    classifier = Classifier(seed=seed_parameter).to(rank)
    ddp_classifier = DDP(classifier, device_ids=list(range(world_size)), output_device=rank, find_unused_parameters=True)
    optimizer = optimizer(ddp_classifier.parameters(), lr=learning_rate_init)
    scheduler = MultiStepLR(optimizer, milestones=[292, 373], gamma=gamma)
    for epoch in range(epochs):
        data_loader_train.sampler.set_epoch(epoch)
        ddp_classifier.train()
        for image, label in data_loader_train:
            image = image.to(rank)
            label = label.to(rank)
            optimizer.zero_grad(set_to_none=True)
            prediction = ddp_classifier(image, train=True)
            loss_train = mse(prediction, label)
            loss_train.mean().backward()
            torch.nn.utils.clip_grad_norm_(ddp_classifier.parameters(), max_norm=1)
            optimizer.step()
        if rank == 0:
            accs_train = measure_accuracy_classifier(prediction, label)
            ddp_classifier.eval()
            loss_valid = 0
            accs_valid = Counter({group:0 for group in range(1,12)})
            with torch.no_grad():
                for N_test, (image, label) in enumerate(data_loader_valid):
                    image = image.to(rank)
                    label = label.to(rank)
                    prediction = ddp_classifier(image, train=False)
                    loss_valid += mse(predction, label).item()
                    accs_valid.update(measure_accuracy_classifier(prediction, label))
            for group in accs.keys():
                accs[group] /= N_test + 1
            logs = {"loss_train": np.sqrt(loss_train.item()),
                    "loss_valid": np.sqrt(loss_valid)}
            logs.update({f"accuracy_Q{group}_train":acc for group, acc in accs_train.items()})
            logs.update({f"accuracy_Q{group}_valid":acc for group, acc in accs_valid.items()})
            wandb.log(logs)
    # safe full model for later use
    if not epoch % 100:
        torch.save(ddp_classifier.model, folder_results + f"classifier_model_epoch{epoch}.pth")
    # end distributed process
    dist.destroy_process_group()
    


def run_train_classifier_distributed():
    wandb.login(key="834835ffb309d5b1618c537d20d23794b271a208")
    wandb.init(**wandb_kwargs)
    world_size = N_gpus
    print("N_gpus", world_size)
    world_size = 8
    mp.spawn(train_classifier_distributed,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    wandb.finish()

def load_ddp_model_to_non_ddp_model():
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model.load_state_dict(model_dict)

    
if __name__ == "__main__":

    run_train_classifier_distributed()
    
#    train_classifier_on_hyperparameters(seed_parameter=seed_parameter)


#    train_classifier_on_random_hyperparameters(seed_parameter=seed_parameter)
