from functools import partial

import torch

from labeling import class_groups_indices


def measure_mse_accuracy_classifier_group(labels_prediction: torch.Tensor,
                                          labels_target: torch.Tensor,
                                          minimum_answers=0.5,
                                          allowed_deviation=0.1
                                         ) -> float:
    """
    measure accuracy of classifier prediction. Calculates MSE loss on single label group
    where at least miminum_answers of people answered this question
    
    Parameters
    ----------
    labels_prediction(torch.tensor) : labels predicted by classifier
    labels_target(torch.tensor) : original labels
    minimum_answer(float) : (0.0 to 1.0) minimum amount of collected answers on a question to be considered
    """
    consider = torch.sum(labels_target, dim=1) > minimum_answers
    deviation = torch.abs(labels_prediction[consider] / labels_target[consider] - 1)
    nans = torch.isnan(deviation)
    deviation[nans] = torch.abs(labels_prediction[consider][nans] - labels_target[consider][nans])
    accuracy = torch.mean((deviation < allowed_deviation).float()).item()
    return accuracy


def measure_accuracy_classifier_group(labels_prediction: torch.Tensor,
                                      labels_target: torch.Tensor,
                                      minimum_answers=0.5,
                                      allowed_deviation=0.1
                                     ) -> float:
    """
    measure accuracy of classifier prediction for individual group. 
    for questions answered by at least miminum_answers of people
    return relative amount of agreements for maxed label
    
    Parameters
    ----------
    labels_prediction(torch.tensor) : labels predicted by classifier
    labels_target(torch.tensor) : original labels
    minimum_answer(float) : (0.0 to 1.0) minimum amount of collected answers on a question to be considered
    """
    consider = torch.sum(labels_target, dim=1) > minimum_answers
    pred = torch.argmax(labels_prediction[consider], dim=1)
    targ = torch.argmax(labels_target[consider], dim=1)
    accuracy = torch.mean((pred == targ).float()).item()
    return accuracy

def measure_accuracy_classifier(labels_prediction: torch.Tensor,
                                labels_target: torch.Tensor,
                                minimum_answers=0.5,
                                allowed_deviation=0.1, considered_groups=range(1,12)
                               ) -> dict:
    """
    measure accuracy of classifier prediction. Calculates accuracy on label groups
    where at least miminum_answers of people answered this question
    returns loss for each group
    
    Parameters
    ----------
    labels_prediction(torch.tensor) : labels predicted by classifier
    labels_target(torch.tensor) : original labels
    minimum_answer(float) : (0.0 to 1.0) minimum amount of collected answers on a question to be considered
    """
    measure_accuracy = partial(measure_accuracy_classifier_group,
                               minimum_answers=minimum_answers,
                               allowed_deviation=allowed_deviation)
    accuracies = {group:measure_accuracy(labels_prediction[:,ix], labels_target[:,ix])
                  for group, ix in class_groups_indices.items()
                  if group in considered_groups}
    return accuracies


def accuracy_discriminator(target: torch.Tensor, prediction: torch.Tensor, p=0.9) -> float:
    """ measure accuracy of discriminator results
    count predictions with p or less certainty for correct target
    """
    if prediction is 0:
        return 0
    accurate = torch.abs(target - prediction) < 1 - p
    return torch.mean(accurate.float()).item()


def accuracy_classifier(target: torch.Tensor, prediction: torch.Tensor, p=0.1) -> float:
    """ measure accuracy of classifier results
    count predictions with all classes at p or less distance to target
    """
    if prediction is 0:
        return 0
    accurate = torch.abs(target - prediction) < p
    accurate = torch.all(accurate, dim=1)
    return torch.mean(accurate.float()).item()
