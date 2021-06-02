import numpy as np
from torch import cat


label_group_sizes = [3,2,2,2,4,2,3,7,3,3,6]
labels_dim = np.sum(label_group_sizes)

class_groups = {
    0 : (),
    1 : (1,2,3),
    2 : (4,5),
    3 : (6,7),
    4 : (8,9),
    5 : (10,11,12,13),
    6 : (14,15),
    7 : (16,17,18),
    8 : (19,20,21,22,23,24,25),
    9 : (26,27,28),
    10 : (29,30,31),
    11 : (32,33,34,35,36,37),
}

class_groups_indices = {g:np.array(ixs) for g, ixs in class_groups.items()}


def make_galaxy_labels_hierarchical(groups):
    """ transform groups of galaxy label probabilities to follow the hierarchical order defined in galaxy zoo
    more info here: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview/the-galaxy-zoo-decision-tree
    groups is a list of Nxl torch tensors, where N is the batch size
    and l is the number of labels in each group, listed in label_group_sizes
    in each group probabilities should add to one

    Return
    ------
    hierarchical_labels : NxL torch tensor, where L is the total number of labels
    """
    ## first [0] group has norm unity
    groups[1] = groups[1] * groups[0][:,1].unsqueeze(-1) ## edge on
    groups[2] = groups[2] * groups[1][:,1].unsqueeze(-1) ## sign of a bar
    groups[3] = groups[3] * groups[1][:,1].unsqueeze(-1) ## sign of spirals
    groups[4] = groups[4] * groups[1][:,1].unsqueeze(-1) ## bulge prominence
#    groups[5] = groups[5] * sum(groups[0][:,:2], dim=1).unsqueeze(-1) ## anything odd  ## should be nomalized without artifact (1,3)
    groups[6] = groups[6] * groups[0][:,0].unsqueeze(-1) ## how round
    groups[7] = groups[7] * groups[5][:,0].unsqueeze(-1) ## odd features
    groups[8] = groups[8] * groups[1][:,0].unsqueeze(-1) ## bulge shape
    groups[9] = groups[9] * groups[3][:,0].unsqueeze(-1) ## tightly wound arms
    groups[10] = groups[10] * groups[3][:,0].unsqueeze(-1) ## how many arms
    hierarchical_labels = cat(groups, dim=1)
    return hierarchical_labels
