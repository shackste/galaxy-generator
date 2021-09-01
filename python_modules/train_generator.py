import torch
from tqdm import trange, tqdm

from dataset import MakeDataLoader
from labeling import generate_labels, ConsiderGroups
from sampler import generate_latent
from image_classifier import ImageClassifier
from big.BigGAN2 import Generator, Discriminator
from big.losses import discriminator_loss, generator_loss

considered_groups = [1, 2, 3, 4, 5, 6]
consider_groups = ConsiderGroups(considered_groups=considered_groups)
considered_label_indices = consider_groups.get_considered_labels()

labels_dim = consider_groups.get_labels_dim() ## 37
latent_dim = 128

N_dis_train = 2 # How many times discriminator is trained before generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generator = Generator(dim_z=latent_dim, labels_dim=labels_dim).to(device)
discriminator = Discriminator(labels_dim=labels_dim).to(device)
classifier = ImageClassifier().to(device)
classifier.load()  ## use pretrained classifier

loss_class = torch.nn.MSELoss()




def train_step(images: torch.Tensor, labels: torch.Tensor, train_generator: bool=True):
    latent = generate_latent(labels.shape[0], latent_dim, sigma=False)
    labels_fake = generate_labels(labels.shape[0])
    label_transformed_fake_gen = generator.transform_labels(labels_fake[:,considered_label_indices])
    generated_images = generator(latent, label_transformed_fake_gen)

    # disc training
    label_transformed_fake_dis = discriminator.transform_labels(labels_fake[:,considered_label_indices])
    label_transformed_real_dis = discriminator.transform_labels(labels[:,considered_label_indices])

    prediction_fake = discriminator(generated_images.detach(), label_transformed_fake_dis).view(-1)
    prediction_real = discriminator(images, label_transformed_real_dis).view(-1)

    d_loss_real, d_loss_fake = discriminator_loss(prediction_fake, prediction_real)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()

    discriminator.optim.step()

    # gen training
    if train_generator:
        prediction = discriminator(generated_images, label_transformed_fake_dis.detach()).view(-1)
        labels_prediction = classifier.predict(generated_images)

        g_loss = generator_loss(prediction)
        g_loss += loss_class(labels_prediction[:,considered_label_indices], labels[:,considered_label_indices])

        g_loss.backward()

        generator.optim.step()


def train_epoch(data_loader, epoch: int=0):
    i = 0
    for images, labels in tqdm(data_loader, desc=f"epoch {epoch}"):
        train_step(images, labels, train_generator=bool(i % N_dis_train))
        i += 1


def train_biggan(batch_size: int = 1, epochs: int = 2):
    make_data_loader = MakeDataLoader()
    for epoch in trange(epochs, desc="epochs"):
        data_loader_train = make_data_loader.get_data_loader_train(batch_size=batch_size)
        train_epoch(data_loader_train, epoch)
        generator.save()

        ## evaluation, save losses to wandb
        '''
        data_loader_valid = make_data_loader.get_data_loader_valid(batch_size=batch_size)

        for images, labels in data_loader_train:
            break
        evaluate(images, labels)
        '''

if __name__ == "__main__":
    train_biggan()