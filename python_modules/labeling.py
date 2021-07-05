import numpy as np
from torch import cat, sum


label_group_sizes = [3,2,2,2,4,2,3,7,3,3,6]
labels_dim = np.sum(label_group_sizes)

class_groups = {
    # group : indices
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

hierarchy = {
    # group : parent (group, label)
    2: (1, 1),
    3: (2, 1),
    4: (2, 1),
    5: (2, 1),
    7: (1, 0),
    8: (6, 0),
    9: (2, 0),
    10: (4, 0),
    11: (4, 0),
}



def make_galaxy_labels_hierarchical(labels):
    """ transform groups of galaxy label probabilities to follow the hierarchical order defined in galaxy zoo
    more info here: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview/the-galaxy-zoo-decision-tree
    labels is a NxL torch tensor, where N is the batch size and L is the number of labels,
    all labels should be > 1
    the indices of label groups are listed in class_groups_indices

    Return
    ------
    hierarchical_labels : NxL torch tensor, where L is the total number of labels
    """
    groups = lambda i: labels[:,class_groups_indices[i]]

    for i in range(1,12):
        group = groups(i)
        ## normalize probabilities to 1
        group = group / sum(group, dim=1)
        ## follow hierarchical structure
        if i not in [1,6]:
            group = group * groups(hierarchy[i][0])[:,hierarchy[i][1]].unsqueeze(-1)
    return labels
