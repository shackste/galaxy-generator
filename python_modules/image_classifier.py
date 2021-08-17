from collections import Counter
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Sequential, Linear, Conv2d, MaxPool2d, Dropout, ReLU, Flatten, LeakyReLU
import torchvision.models as models
from torchvision.transforms import Compose, FiveCrop, Lambda, RandomErasing
from torchvision.transforms.functional import rotate, hflip
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from neuralnetwork import NeuralNetwork, update_networks_on_loss
from additional_layers import MaxOut, Conv2dUntiedBias, ALReLU
from labeling import make_galaxy_labels_hierarchical, label_group_sizes, labels_dim, class_group_layers, class_groups_indices
from loss import mse, rmse, loss_sample_variance, get_sample_variance, plot_losses
from accuracy_measures import measure_accuracy_classifier
from file_system import folder_results
from losses import Losses, Accuracies



class ClassifierBase(NeuralNetwork):
    def __init__(self,
                 considered_layers_init = [1, 2, 3, 4, 5][:4],  ## group layers to be considered from start
                ):
        super(ClassifierBase, self).__init__()
        
        self.considered_groups = set() ## container for set of groups to be considered in training and evaluation
        self.considered_label_indices = []
        for layer in considered_layers_init:
            self.consider_layer(layer)
        try:
            self.considered_groups.remove(9) ## Question 9 is not modeled well
        except:
            pass
        self.update_considered_label_indices()
        
    def consider_groups(self, groups: set) -> None:
        """ add group to considered_groups """
        self.considered_groups.update(groups)

    def consider_layer(self, layer: int) -> None:
        """ add groups in layer to considered_groups """
        self.consider_groups(class_group_layers[layer])

    def update_considered_label_indices(self) -> None:
        """ find label indices of groups to be considered """
        for group, indices in class_groups_indices.items():
            if group in self.considered_groups:
                for ix in indices:
                    if not ix in self.considered_label_indices:
                        self.considered_label_indices.append(ix)
        self.considered_label_indices.sort()

    def add_considered_layer(self, layer: int) -> None:
        """ add layer to considered layers and update considered label indices """
        self.consider_layer(layer)
        self.update_considered_label_indices()



class ImageClassifier(ClassifierBase):
    def __init__(self, 
                 seed=None,
                 optimizer=SGD, optimizer_kwargs = {"nesterov":True, "momentum":0.9},
                 learning_rate_init = 0.04,
                 gamma = 0.995, # learning rate decay factor
                 considered_layers_init = [1, 2, 3, 4, 5][:4],  ## group layers to be considered from start
                 sample_variance_threshold = 0.002,
                 weight_loss_sample_variance = 0, # 10.
                 evaluation_steps = 250 # number of batches between loss tracking

                ):
        super(ImageClassifier, self).__init__(considered_layers_init=considered_layers_init)
        if seed is not None:
            torch.manual_seed(seed)

        #'''
        resnet = models.resnet18(pretrained=False)
        self.conv = Sequential(
            *(list(resnet.children())[:-1]),
            Flatten(),
        )
        '''  architecture used by Dielemann et al 2015
        self.conv = Sequential(
#            Conv2dUntiedBias(41, 41, 3, 32, kernel_size=6),
            Conv2d(3,32, kernel_size=6),
            ReLU(),
            MaxPool2d(2),
#            Conv2dUntiedBias(16, 16, 32, 64, kernel_size=5),
            Conv2d(32, 64, kernel_size=5),
            ReLU(),
            MaxPool2d(2),
#            Conv2dUntiedBias(6, 6, 64, 128, kernel_size=3),
            Conv2d(64, 128, kernel_size=3),
            ReLU(),
#            Conv2dUntiedBias(4, 4, 128, 128, kernel_size=3), #weight_std=0.1),
            Conv2d(128, 128, kernel_size=3),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
        )
        #'''
        self.dense1 = MaxOut(8192, 2048, bias=0.01) 
        self.dense2 = MaxOut(2048, 2048, bias=0.01) 
        self.dense3 = Sequential(
            MaxOut(2048, 37, bias=0.1),
#            LeakyReLU(negative_slope=1e-7),
            ALReLU(negative_slope=1e-2),
        )
        self.dropout = Dropout(p=0.5)
        
        self.augment = Compose([
            Lambda(lambda img: torch.cat([img, hflip(img)], 0)),
            Lambda(lambda img: torch.cat([img, rotate(img, 45)], 0)),
            FiveCrop(45),
            Lambda(lambda crops: torch.cat([rotate(crop, ang) for crop, ang in zip(crops, (0, 90, 270, 180))], 0)),
        ])
        self.random_eraser = RandomErasing(scale=(0.01, 0.2))
        self.N_augmentations = 16
        self.N_conv_outputs = 512

        self.set_optimizer(optimizer, lr=learning_rate_init, **optimizer_kwargs)
#        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[292, 373], gamma=gamma)

        self.make_labels_hierarchical = False # if True, output probabilities are renormalized to fit the hierarchical label structure
        self.N_batches_test = 1
        self.evaluation_steps = evaluation_steps # number of batches between loss tracking
        self.weight_loss_sample_variance = weight_loss_sample_variance
        self.sample_variance_threshold = sample_variance_threshold
        
        self.iteration = 0
        self.epoch = 0
        self.losses_train = Losses("loss", "train")
        self.losses_valid = Losses("loss", "valid")
        self.sample_variances_train = Losses("sample variance", "train")
        self.sample_variances_valid = Losses("sample variance", "valid")
        for g in range(1,12):
            setattr(self, f"accuracies_Q{g}_train", Accuracies("accuracy train", f"Q{g}"))
            setattr(self, f"accuracies_Q{g}_valid", Accuracies("accuracy valid", f"Q{g}"))
        self.losses_regression = Losses("loss", "regression")
        self.losses_variance = Losses("loss", "sample variance")

        ## return to random seed
        if seed is not None:
            sd = np.random.random()*10000
            torch.manual_seed(sd)


    def update_optimizer(self, **kwargs) -> None:
        self.set_optimizer(optimizer, **kwargs)
        
    def update_optimizer_learningrate(self, learning_rate) -> None:
        print("update lr", learning_rate)
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = learning_rate
        
    def use_label_hierarchy(self) -> None:
        self.make_labels_hierarchical = True
        
    def forward(self, x: torch.Tensor, train=False) -> torch.Tensor:
        if train and False:
            x = self.random_eraser(x)
        x = self.augment(x)
        x = self.conv(x)

        x = self.recombine_augmentation(x)

        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
#        x += 1e-4  ## use only with LeakyReLU to prevent values < 0
        if self.make_labels_hierarchical:
            x = make_galaxy_labels_hierarchical(x)
        return x

    def recombine_augmentation(self, x) -> torch.Tensor:
        """ recombine results of augmented views to single vector """
        batch_size = x.size(0) // self.N_augmentations
        x = x.reshape(self.N_augmentations, batch_size, self.N_conv_outputs)
        x = x.permute(1,0,2)
        x = x.reshape(batch_size, self.N_augmentations*self.N_conv_outputs)
        return x



    def train_step(self, images: torch.tensor, labels: torch.tensor) -> float:
        self.train()
        labels_pred = self.forward(images, train=True)
        loss_regression = mse(labels_pred[:,self.considered_label_indices], labels[:,self.considered_label_indices])
        loss_variance = self.weight_loss_sample_variance * \
            loss_sample_variance(labels_pred[:,self.considered_label_indices], 
                                 threshold=self.sample_variance_threshold)
        loss = loss_regression + loss_variance
        self.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        self.optimizer.step()
        self.iteration += 1
        return loss.item()

    def train_epoch(self, 
                    data_loader_train: torch.utils.data.DataLoader,
                    data_loader_valid: torch.utils.data.DataLoader,
                   ) -> None:
        for images, labels in tqdm(data_loader_train, desc=f"epoch {self.epoch}"):
            loss = self.train_step(images, labels)
            if np.isnan(loss):
                from pdb import set_trace
                set_trace()
                loss = self.train_step(images, labels)
                raise Exception("loss is NaN")
            if not self.iteration % self.evaluation_steps - 1:
                loss_regression, loss_variance, accs, variance = self.evaluate_batch(images, labels, print_labels=True)
                self.losses_regression.append(self.iteration, loss_regression)
                self.losses_variance.append(self.iteration, loss_variance)
                self.losses_train.append(self.iteration, loss_regression+loss_variance)
                self.sample_variances_train.append(self.iteration, variance)
                for group, acc in accs.items():
                    getattr(self, f"accuracies_Q{group}_train").append(self.iteration, acc)
                for images, labels in data_loader_valid:
                    break
                loss_regression, loss_variance, accs, variance = self.evaluate_batch(images, labels)
                self.losses_valid.append(self.iteration, loss_regression+loss_variance)
                self.sample_variances_valid.append(self.iteration, variance)
                for group, acc in accs.items():
                    getattr(self, f"accuracies_Q{group}_valid").append(self.iteration, acc)

        self.epoch += 1
        self.scheduler.step()
        self.save()
        
    def evaluate_batches(self, data_loader: torch.utils.data.DataLoader) -> list:
        with torch.no_grad():
            loss = 0
            accs = Counter({group:0 for group in range(1,12)})
            variance = 0
            for N_test, (images, labels) in enumerate(data_loader):
                if N_test >= self.N_batches_test:
                    break
                loss_, accs_, variance_ = self.evaluate_batch(images, labels)
                loss += loss_
                accs.update(accs_)
                variance += varance_
            loss /= self.N_batches_test
            variance /= self.N_batches_test
            for group in accs.keys():
                accs[group] /= self.N_batches_test
        return loss, accs, variance

        
    def evaluate_batch(self, images: torch.tensor, labels: torch.tensor, print_labels=False) -> tuple:
        """ evaluations for batch """
        self.eval()
        with torch.no_grad():
            labels_pred = self.forward(images)
            if print_labels:
                for i, (prediction, target) in enumerate(zip(labels_pred, labels)):
                    print("target\t\t", np.around(target[self.considered_label_indices].cpu(),3))
                    print("\033[1mprediction\t", np.around(prediction[self.considered_label_indices].cpu(),3), end="\033[0m\n")
                    if i >= 2:
                        break
                print("<target>\t", np.around(torch.mean(labels[:,self.considered_label_indices], dim=0).cpu(), 3))
                print("<target>\t", np.around(torch.std(labels[:,self.considered_label_indices], dim=0).cpu(), 3))
                print("\033[1m<prediction>\t", np.around(torch.mean(labels_pred[:,self.considered_label_indices], dim=0).cpu(), 3), end="\033[0m\n")
                print("\033[1m<prediction>\t", np.around(torch.std(labels_pred[:, self.considered_label_indices], dim=0).cpu(), 3), end="\033[0m\n")
            loss_regression = torch.sqrt(mse(labels_pred[:,self.considered_label_indices],
                                  labels[:,self.considered_label_indices]
                                 )).item()
            loss_variance = self.weight_loss_sample_variance * \
                    loss_sample_variance(labels_pred[:,self.considered_label_indices], 
                                         threshold=self.sample_variance_threshold
                                        ).item()
            accs = measure_accuracy_classifier(labels_pred, labels, considered_groups=self.considered_groups)
            variance = get_sample_variance(labels_pred[:,self.considered_label_indices]).item()
        return loss_regression, loss_variance, accs, variance

    
    def plot_losses(self, save=False):
        self.losses_train.plot()
        self.losses_valid.plot()
        self.losses_regression.plot(linestyle=":")
        self.losses_variance.plot(linestyle=":")
        if save:
            plt.savefig(folder_results+"loss.png")
            plt.close()
        else:
            plt.show()

    def plot_sample_variances(self, save=False):
        self.sample_variances_train.plot()
        self.sample_variances_valid.plot()
        if save:
            plt.savefig(folder_results+"variances.png")
            plt.close()
        else:
            plt.show()

    def plot_accuracy(self, save=False):
        for group in range(1,12):
            if not group in self.considered_groups:
                continue
            getattr(self, f"accuracies_Q{group}_train").plot()
        if save:
            plt.savefig(folder_results+"accuracy_train.png")
            plt.close()
        else:
            plt.show()
        
    def plot_test_accuracy(self, save=False):
        for group in range(1,12):
            if not group in self.considered_groups:
                continue
            getattr(self, f"accuracies_Q{group}_valid").plot()
        if save:
            plt.savefig(folder_results+"accuracy_valid.png")
            plt.close()
        else:
            plt.show()