import numpy as np
import torch
from torch import cat, sum, Tensor


label_group_sizes = [3,2,2,2,4,2,3,7,3,3,6]
labels_dim = np.sum(label_group_sizes)

class_groups = {
    # group : indices (assuming 0th position is id)
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

class_group_layers = {
    1 : [1, 6],
    2 : [2, 7],
    3 : [3, 9],
    4 : [4, 5],
    5 : [8, 10, 11],
}

class_groups_indices = {g:np.array(ixs)-1 for g, ixs in class_groups.items()}
#class_groups_indices = {g:Tensor(ixs) for g, ixs in class_groups.items()}

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

label_info = { # (name, next group), 0 == end
    ## 1: smooth, disk or artifact/star
    1 : ("smooth",7),
    2 : ("disk",2),
    3 : ("artifact",0),

    ## 2: edge on
    4 : ("edge-on disk",9),
    5 : ("not edge-on",3),

    ## 3: barred
    6 : ("barred",4),
    7 : ("not barred",4),

    ## 4: spiral arms
    8 : ("spiral arms",10),
    9 : ("no spiral arms",5),

    ## 5: bulge
    10 : ("no bulge",6),
    11 : ("noticable bulge",6),
    12 : ("obvios bulge",6),
    13 : ("domintat bulge",6),

    ## 6: anything odd
    14 : ("odd",8),
    15 : ("not odd",0),

    ## 7: roundness
    16 : ("completely round",6),
    17 : ("ellptic",6),
    18 : ("cigar-shaped",6),

    ## 8: odd features
    19 : ("ring",0),
    20 : ("lens",0),
    21 : ("disturbed",0),
    22 : ("irregular",0),
    23 : ("other",0),
    24 : ("merger",0),
    25 : ("dust lane",0),

    ## 9: bulge shape
    26 : ("rounded",6),
    27 : ("boxy",6),
    28 : ("no bulge",6),

    ## 10: tightness of spiral arms
    29 : ("tight",11),
    30 : ("medium",11),
    31 : ("loose",11),

    ## 11: number of spiral arms
    32 : ("one",5),
    33 : ("two",5),
    34 : ("three",5),
    35 : ("four",5),
    36 : ("five+",5),
    37 : ("can't tell",5),
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class ConsiderGroups:
    def __init__(self,
                 considered_groups: list = list(range(12)),  ## groups to be considered from start
                 ):
        self.considered_groups = []
        self.considered_label_indices = []
        for group in considered_groups:
            self.consider_group(group)

    def consider_group(self, group: int) -> None:
        """ add group to considered_label_indices """
        if group in self.considered_groups:
            print(f"group {group} already considered")
            return;
        self.considered_groups.append(group)
        self.considered_label_indices.extend(class_groups_indices[group])
        self.considered_label_indices.sort()

    def unconsider_group(self, group: int) -> None:
        """ add group to considered_label_indices """
        if group not in self.considered_groups:
            print(f"group {group} not considered")
            return;
        self.considered_groups.remove(group)
        for label in class_group_indices[group]:
            self.considered_label_indices.remove(label)
        self.considered_label_indices.sort()

    def get_considered_labels(self) -> list:
        """ returns list of considered label indices """
        return self.considered_label_indices

    def get_labels_dim(self):
        """ obtain dimensions of label vector for considered groups """
        return len(self.considered_label_indices)


def make_galaxy_labels_hierarchical(labels: torch.Tensor) -> torch.Tensor:
    """ transform groups of galaxy label probabilities to follow the hierarchical order defined in galaxy zoo
    more info here: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview/the-galaxy-zoo-decision-tree
    labels is a NxL torch tensor, where N is the batch size and L is the number of labels,
    all labels should be > 1
    the indices of label groups are listed in class_groups_indices

    Return
    ------
    hierarchical_labels : NxL torch tensor, where L is the total number of labels
    """
    shift = labels.shape[1] > 37 ## in case the id is included at 0th position, shift indices accordingly
    index = lambda i: class_groups_indices[i] + shift

    for i in range(1,12):
        ## normalize probabilities to 1
        norm = sum(labels[:,index(i)], dim=1, keepdims=True)
        norm[norm == 0] += 1e-4  ## add small number to prevent NaNs dividing by zero, yet keep track of gradient
        labels[:,index(i)] /= norm
        ## renormalize according to hierarchical structure
        if i not in [1,6]:
            parent_group_label = labels[:,index(hierarchy[i][0])]
            labels[:,index(i)] *= parent_group_label[:,hierarchy[i][1]].unsqueeze(-1)
    return labels


def check_labels_hierarchical(labels: torch.Tensor) -> torch.Tensor:
    """ check whether labels follow correct hierarchy """
    correct = torch.ones(labels.shape[0], dtype=torch.bool)
    for g, ix in class_groups_indices.items():
        if g == 0:
            continue
        if g in [1, 6]:
            group_norm = 1.
        else:
            parent_group, parent_index = hierarchy[g]
            parent_group_labels = labels[:, class_groups_indices[parent_group]]
            group_norm = parent_group_labels[:, parent_index]
        norm = torch.sum(labels[:, ix], dim=1)
        check = torch.round((norm - group_norm) * 4) != 0
        correct[check] = False
    return correct

def generate_labels(batch_size: int):
    """ generate batch of fake random galaxy labels that follow the hierarchical label structure """
    labels = torch.rand(batch_size, labels_dim)
    labels = make_galaxy_labels_hierarchical(labels)
    return labels.to(device)