import torch
from tqdm import trange, tqdm
from functools import partial

from dataset import MakeDataLoader
from labeling import generate_labels, ConsiderGroups
from sampler import generate_latent
from image_classifier import ImageClassifier
from loss import get_sample_variance
from helpful_functions import write_generated_galaxy_images_iteration
from statistical_tests import compute_divergence_total_intensity_statistics, compute_divergence_bluriness_statistics
from big.BigGAN2 import Generator, Discriminator
from big.losses import discriminator_loss, generator_loss

considered_groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
consider_groups = ConsiderGroups(considered_groups=considered_groups)
considered_label_indices = consider_groups.get_considered_labels()

labels_dim = consider_groups.get_labels_dim() ## 37
latent_dim = 128

epochs = 500
batch_size = 1
N_batches_evaluate = 16
N_dis_train = 2 # How many times discriminator is trained before generator

track = False ## if True, track evaluation measures with wandb
plot_images = True

lr_generator = 5e-5
lr_discriminator = 2e-4

hyperparameter_dict = {
    "lr_generator" : lr_generator,
    "lr_discriminator" : lr_discriminator,
}

wandb_kwargs = {
    "project" : "galaxy generator", ## top level identifier
    "group" : "parameter search", ## secondary identifier
    "job_type" : "training", ## third level identifier
    "tags" : ["training", "parameter search"],  ## tags for organizing tasks
    "name" : f"lr_G {lr_generator}, lr_D {lr_discriminator}", ## bottom level identifier, label of graph in UI
    "config" : hyperparameter_dict, ## dictionary of used hyperparameters
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(dim_z=latent_dim, labels_dim=labels_dim, G_lr=lr_generator).to(device)
discriminator = Discriminator(labels_dim=labels_dim, D_lr=lr_discriminator).to(device)
classifier = ImageClassifier().to(device)
classifier.load()  ## use pretrained classifier
classifier.eval() ## clasifier is not trained, but always in evaluation mode

loss_class = torch.nn.MSELoss()



def train_step(images: torch.Tensor, labels: torch.Tensor, train_generator: bool=True):
    latent = generate_latent(labels.shape[0], latent_dim, sigma=False)
    labels_fake = generate_labels(labels.shape[0])
    label_transformed_fake_gen = generator.transform_labels(labels_fake[:,considered_label_indices])
    generated_images = generator(latent, label_transformed_fake_gen)

    # disc training
    d_loss_real, d_loss_fake = compute_loss_discriminator(images, labels, generated_images, labels_fake)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()

    discriminator.optim.step()

    # gen training
    if train_generator:
        g_loss_dis, g_loss_class = compute_loss_generator(generated_images, labels_fake)
        g_loss = g_loss_dis
        g_loss += g_loss_class
        g_loss.backward()

        generator.optim.step()


def train_epoch(data_loader, epoch: int=0):
    generator.train()
    discriminator.train()
    i = 0
    for images, labels in tqdm(data_loader, desc=f"epoch {epoch}"):
        train_step(images, labels, train_generator=bool(i % N_dis_train))
        i += 1
        if i > 4:
            break


def train_biggan(batch_size: int = batch_size, epochs: int = epochs, N_batches_evaluate: int = N_batches_evaluate, plot_images: bool = plot_images):
    make_data_loader = MakeDataLoader()
    for epoch in trange(epochs, desc="epochs"):
        data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size)
        train_epoch(data_loader_train, epoch)
        generator.save()
        discriminator.save()

        ## evaluation, save losses to wandb

        data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size*N_batches_evaluate)
        data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size*N_batches_evaluate)

        for images_train, labels_train in data_loader_train:
            break
        for images_valid, labels_valid in data_loader_valid:
            break
        evaluate(images_train, labels_train, images_valid, labels_valid, track=track, plot_images=plot_images, iteration=epoch)

def compute_loss_discriminator(images: torch.Tensor, labels: torch.Tensor, generated_images: torch.Tensor, labels_fake: torch.Tensor) -> tuple:
    """ compute loss of discriminator """

    label_transformed_fake_dis = discriminator.transform_labels(labels_fake[:, considered_label_indices])
    label_transformed_real_dis = discriminator.transform_labels(labels[:, considered_label_indices])

    prediction_fake = discriminator(generated_images.detach(), label_transformed_fake_dis).view(-1)
    prediction_real = discriminator(images, label_transformed_real_dis).view(-1)

    d_loss_real, d_loss_fake = discriminator_loss(prediction_fake, prediction_real)
    return d_loss_real, d_loss_fake

def compute_loss_generator(generated_images: torch.Tensor, labels: torch.Tensor,):
    """ compute loss of generator """
    label_transformed_dis = discriminator.transform_labels(labels[:, considered_label_indices])
    prediction = discriminator(generated_images, label_transformed_dis.detach()).view(-1)
    labels_prediction = classifier.predict(generated_images)

    g_loss_dis = generator_loss(prediction)
    g_loss_class = loss_class(labels_prediction[:, considered_label_indices], labels[:, considered_label_indices])
    return g_loss_dis, g_loss_class

def evaluate(images_train: torch.Tensor, labels_train: torch.Tensor, images_valid: torch.Tensor, labels_valid: torch.Tensor, track=False, plot_images=False, iteration: int = 0) -> None:
    """ evaluate BigGAN model """
    with torch.no_grad():
        generator.eval()
        discriminator.eval()
        latent = generate_latent(labels_train.shape[0], latent_dim, sigma=False)
#        labels_fake = generate_labels(labels_train.shape[0])
        labels_fake = labels_valid  ## generated images should fit the distribution of validation data. both are checked against training data
        label_transformed_fake_gen = generator.transform_labels(labels_fake[:, considered_label_indices])
        generated_images = generator(latent, label_transformed_fake_gen)
        if plot_images:
            width = 8
            write_generated_galaxy_images_iteration(iteration=iteration, images=generated_images.cpu(), width=width, height=len(generated_images)//width)

        d_loss_real_train, d_loss_fake = compute_loss_discriminator(images_train, labels_train, generated_images, labels_fake)
        d_loss_real_valid, d_loss_fake = compute_loss_discriminator(images_valid, labels_valid, generated_images, labels_fake)

        g_loss_dis, g_loss_class = compute_loss_generator(generated_images, labels_fake)

        variance_train = get_sample_variance(images_train)
        variance_generator = get_sample_variance(generated_images)

#        divergence_intensity_valid = compute_divergence_total_intensity_statistics(images_train, images_valid)
#        divergence_intensity_generator = compute_divergence_total_intensity_statistics(images_train, generated_images)
        divergence_bluriness_valid = compute_divergence_bluriness_statistics(images_train, images_valid)
        divergence_bluriness_generator = compute_divergence_bluriness_statistics(images_train, generated_images)

        logs = {
            "loss_discriminator_real_train": d_loss_real_train.item(),
            "loss_discriminator_real_valid": d_loss_real_valid.item(),
            "loss_discriminator_fake": d_loss_fake.item(),
            "loss_generator_dis": g_loss_dis.item(),
            "loss_generator_class": g_loss_class.item(),
            "variance_train" : variance_train.item(),
            "variance_generator" : variance_generator.item(),
#            "divergence_intensity_valid" : divergence_intensity_valid.item(),
#            "divergence_intensity_generator": divergence_intensity_generator.item(),
            "divergence_bluriness_valid": divergence_bluriness_valid.item(),
            "divergence_bluriness_generator": divergence_bluriness_generator.item(),
        }

        if track:
            import wandb
            wandb.log(logs)
        else:
            print(logs)

def train_biggan_tracked(*args, wandb_kwargs: dict = wandb_kwargs, **kwargs):
    from track_progress import track_progress
    train = partial(train_biggan, *args, **kwargs)
    track_progress(train, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    if track:
        train_biggan_tracked(epochs=epochs)
    else:
        train_biggan(epochs=epochs)