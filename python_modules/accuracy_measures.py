from torch import abs, mean, all

def accuracy_discriminator(target, prediction, p=0.9):
    """ measure accuracy of discriminator results
    count predictions with p or less certainty for correct target
    """
    if prediction is 0:
        return 0
    accurate = abs(target - prediction) < 1 - p
    return mean(accurate.float()).item()


def accuracy_classifier(target, prediction, p=0.1):
    """ measure accuracy of classifier results
    count predictions with all classes at p or less distance to target
    """
    if prediction is 0:
        return 0
    accurate = abs(target - prediction) < p
    accurate = all(accurate, dim=1)
    return mean(accurate.float()).item()
