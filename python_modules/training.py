from functools import partial
import collections
import json

from tqdm import trange, tqdm
import torch

from dataset import MakeDataLoader, DataLoader
from labeling import generate_labels, ConsiderGroups
from sampler import generate_latent
from image_classifier import ImageClassifier
from helpful_functions import write_generated_galaxy_images_iteration

use_gan = "vanilla"
use_gan = "BigGAN"

if use_gan == "vanilla":
    from vanilla_gan.gan import Generator, Discriminator, generator_loss, discriminator_loss
elif use_gan == "BigGAN":
    from big.BigGAN2 import Generator, Discriminator
    from big.losses import discriminator_loss, generator_loss


# training parameters
considered_groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # try less groups ## try feeding one-hot labels ## try using simpler dataset (cifar-10, mnist ...)
consider_groups = ConsiderGroups(considered_groups=considered_groups)
considered_label_indices = consider_groups.get_considered_labels()

labels_dim = consider_groups.get_labels_dim()  # 37 for all groups
latent_dim = 128

epochs = 200
mini_batch_size = 4 # 1 * N_sample
batch_size = 64
steps = batch_size // mini_batch_size
N_batches_evaluate = 900
N_dis_train = 2  # How many times discriminator is trained before generator
epochs_evaluation = 1  # number of epochs after which to evaluate
num_workers = 0
pretrain_discriminator = 0  # number of epochs discriminator is trained before generator. Only use together with reload=True

track = False  # if True, track evaluation measures with wandb
plot_images = True
conditional = False  # if True, generator is trained to produce images according to the input labels
reload = False  # if True, continue training

lr_generator = 5e-5  # 1e-3 #5es-5
lr_discriminator = 2e-4  # 1e-4 #2e-4
hyperparameter_dict = {
    "lr_generator" : lr_generator,
    "lr_discriminator" : lr_discriminator,
}

wandb_kwargs = {
    "project" : "galaxy generator",  # top level identifier
    "group" : "probe GANs",  # secondary identifier
    "job_type" : "BigGAN",  # third level identifier
    "tags" : ["training", "parameter search"],  # tags for organizing tasks
    "name" : f"lr_G {lr_generator}, lr_D {lr_discriminator}",  # bottom level identifier, label of graph in UI
    "config" : hyperparameter_dict, # dictionary of used hyperparameters
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

generator = Generator(latent_dim=latent_dim, labels_dim=labels_dim, G_lr=lr_generator).to(device)
discriminator = Discriminator(labels_dim=labels_dim, D_lr=lr_discriminator).to(device)

if reload:
    generator.load()
#    discriminator.load()
if conditional:
    classifier = ImageClassifier().to(device)
    classifier.load()  # use pretrained classifier
    classifier.eval()  # classifier is not trained, but always in evaluation mode
    classifier.use_label_hierarchy()

loss_class = torch.nn.MSELoss()


def train_discriminator(images: torch.Tensor,
                        labels: torch.Tensor,
                        optimizer_step: bool = True):
    latent = generate_latent(labels.shape[0], latent_dim, sigma=False)
    labels_fake = generate_labels(labels.shape[0])
    if not conditional:
        labels_fake[:] = 0
    generated_images = generator(latent, labels_fake)

    d_loss_real, d_loss_fake = compute_loss_discriminator(images, labels, generated_images, labels_fake)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    if optimizer_step:
        discriminator.optimizer.step()
        discriminator.zero_grad()


def train_generator(batch_size: int = mini_batch_size,
                    optimizer_step: bool = True):
    latent = generate_latent(batch_size, latent_dim, sigma=False)
    labels_fake = generate_labels(batch_size)
    if not conditional:
        labels_fake[:] = 0

    generated_images = generator(latent, labels_fake)
    g_loss_dis, g_loss_class = compute_loss_generator(generated_images, labels_fake)
    g_loss = g_loss_dis
    if conditional:
        g_loss += g_loss_class
    g_loss.backward()
    if optimizer_step:
        generator.optimizer.step()
        generator.zero_grad()


def train_epoch(data_loader, epoch: int=0, steps: int = steps):
    """ train generator and discriminator on batch sizes that exceed available memory
        by collecting the backward loss over several (steps) iterations
    """
    generator.train()
    discriminator.train()
    for i, (images, labels) in enumerate(tqdm(data_loader, desc=f"epoch {epoch}")):
        train_discriminator(images, labels, optimizer_step=not (i+1) % steps)
        if epoch < pretrain_discriminator:
            continue
        if not (i+1) % (steps*N_dis_train):
            del images, labels
            for i_gen in range(steps):
                train_generator(batch_size=mini_batch_size, optimizer_step=not (i_gen+1) % steps)


def training(batch_size: int = mini_batch_size,
             steps: int = steps,
             epochs: int = epochs,
             N_batches_evaluate: int = N_batches_evaluate,
             plot_images: bool = plot_images,
             augmented: bool = True):
    torch.backends.cudnn.benchmark = True # use autotuner to find the kernel with best performance
    make_data_loader = MakeDataLoader(augmented=augmented)
    data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size,
                                                               shuffle=True,
                                                               num_workers=num_workers)
    data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size,
                                                               shuffle=True,
                                                               num_workers=num_workers)
    for epoch in trange(epochs, desc="epochs"):
        if not epoch % epochs_evaluation:
            print("evaluate")
            evaluate(data_loader_train, data_loader_valid, track=track, plot_images=plot_images, iteration=epoch)
        train_epoch(data_loader_train, epoch, steps=steps)
        generator.save()
        discriminator.save()


def compute_loss_discriminator(images: torch.Tensor,
                               labels: torch.Tensor,
                               generated_images: torch.Tensor,
                               labels_fake: torch.Tensor,
                               accuracy: bool = False) -> tuple:
    """ compute loss of discriminator """

    prediction_real = discriminator(images, labels).view(-1)
    prediction_fake = discriminator(generated_images.detach(), labels_fake).view(-1)

    d_loss_real, d_loss_fake = discriminator_loss(prediction_fake, prediction_real)
    if not accuracy:
        return d_loss_real, d_loss_fake

#    print("pred real", prediction_real)
#    print("pred fake", prediction_fake)

    accuracy_real = torch.sum(prediction_real > 0) / prediction_real.shape[0]
    accuracy_fake = torch.sum(prediction_fake < 0) / prediction_fake.shape[0]
    return d_loss_real, d_loss_fake, accuracy_real, accuracy_fake


def compute_loss_generator(generated_images: torch.Tensor,
                           labels: torch.Tensor,
                           accuracy: bool = False):
    """ compute loss of generator """
    prediction = discriminator(generated_images, labels.detach()).view(-1)
    g_loss_dis = generator_loss(prediction)
    if not conditional:
        g_loss_class = 0
    else:
        labels_prediction = classifier.predict(generated_images)
        g_loss_class = loss_class(labels_prediction[:, considered_label_indices], labels[:, considered_label_indices])
    if not accuracy:
        return g_loss_dis, g_loss_class

#    print("pred gen", prediction)
    accuracy_dis = torch.sum(prediction > 0) / prediction.shape[0]
    accuracy_class = 0

    return g_loss_dis, g_loss_class, accuracy_dis, accuracy_class


@torch.no_grad()
def evaluate(data_loader_train: DataLoader,
             data_loader_valid: DataLoader,
             track=False,
             plot_images=False,
             iteration: int = 0) -> None:
    """ evaluate BigGAN over full validation dataset """
    full_logs = collections.Counter()
    for i, ((images_train, labels_train), (images_valid, labels_valid)) in enumerate(zip(data_loader_train, data_loader_valid)):
        logs = evaluate_batch(images_train, labels_train, images_valid, labels_valid, plot_images=i==0 and plot_images, iteration=iteration)
        full_logs.update(logs)
        if i >= N_batches_evaluate:
            break
    N = i + 1
    for item in full_logs:
        full_logs[item] /= N
    if track:
        import wandb
        wandb.log(full_logs)
    else:
        print(json.dumps(full_logs, indent=4, sort_keys=True))


@torch.no_grad()
def evaluate_batch(images_train: torch.Tensor,
             labels_train: torch.Tensor,
             images_valid: torch.Tensor,
             labels_valid: torch.Tensor,
             plot_images=False,
             iteration: int = 0) -> dict:
    """ evaluate BigGAN model """
    generator.eval()
    discriminator.eval()
    latent = generate_latent(labels_train.shape[0], latent_dim, sigma=False)
    #        labels_fake = generate_labels(labels_train.shape[0])
    labels_fake = labels_valid  # generated images should fit the distribution of validation data. both are checked against training data
    generated_images = generator(latent, labels_fake)
    if plot_images:
        width = min(8, len(generated_images))
        write_generated_galaxy_images_iteration(iteration=iteration, images=generated_images.cpu(), width=width, height=len(generated_images)//width)
        write_generated_galaxy_images_iteration(iteration=0, images=images_train.cpu(), width=width, height=len(generated_images)//width, file_prefix="original_sample")

    d_loss_real_train, d_loss_fake, accuracy_real_train, accuracy_fake = compute_loss_discriminator(images_train, labels_train, generated_images, labels_fake, accuracy=True)
    d_loss_real_valid, d_loss_fake, accuracy_real_valid, accuracy_fake  = compute_loss_discriminator(images_valid, labels_valid, generated_images, labels_fake, accuracy=True)

    g_loss_dis, g_loss_class, accuracy_gen_dis, accuracy_gen_class = compute_loss_generator(generated_images, labels_fake, accuracy=True)

#    variance_train = get_sample_variance(images_train)
#    variance_generator = get_sample_variance(generated_images)

#        divergence_intensity_valid = compute_divergence_total_intensity_statistics(images_train, images_valid)
#        divergence_intensity_generator = compute_divergence_total_intensity_statistics(images_train, generated_images)
#    divergence_bluriness_valid = compute_divergence_bluriness_statistics(images_train, images_valid)
#    divergence_bluriness_generator = compute_divergence_bluriness_statistics(images_train, generated_images)

    logs = {
        "loss_discriminator_real_train": d_loss_real_train.item(),
        "loss_discriminator_real_valid": d_loss_real_valid.item(),
        "loss_discriminator_fake": d_loss_fake.item(),
        "loss_generator_dis": g_loss_dis.item(),
###            "variance_train" : variance_train.item(),
###            "variance_generator" : variance_generator.item(),
#            "divergence_intensity_valid" : divergence_intensity_valid.item(),
#            "divergence_intensity_generator": divergence_intensity_generator.item(),
###            "divergence_bluriness_valid": divergence_bluriness_valid.item(),
###            "divergence_bluriness_generator": divergence_bluriness_generator.item(),
        "accuracy_real_train" : accuracy_real_train.item(),
        "accuracy_real_valid": accuracy_real_valid.item(),
        "accuracy_fake": accuracy_fake.item(),
        "accuracy_generator_dis": accuracy_gen_dis.item(),
    }
    if conditional:
        logs.update({
            "loss_generator_class" : g_loss_class.item(),
            "accuracy_generator_class": accuracy_gen_class.item(),
        })

    return logs


def training_tracked(*args, wandb_kwargs: dict = wandb_kwargs, **kwargs):
    from track_progress import track_progress
    train = partial(training, *args, **kwargs)
    track_progress(train, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    if track:
        training_tracked()
    else:
        training()