# Here we put procedures for the statistical investigation and comparison
# of morphological properties of sets of galaxy images
import numpy as np
import types
from multiprocessing import Pool

import torch
from corner import corner
from chamferdist import ChamferDistance

from statmorph import get_morphology_measures


measures_groups = {"CAS": ["concentration", "asymmetry", "smoothness", ],
                   "MID": ["multimode", "intensity", "deviation", ],
                   "gini-m20": ["m20", "gini", ],
                   "ellipticity": ["ellipticity_asymmetry", ], }


measures_of_interest = [m
                        for measures in measures_groups.values()
                        for m in measures]

device = "cuda" if torch.cuda.is_available() else "cpu"

# COLLECT DATA FROM IMAGES


def collect_morphology_measures(image):
    """ obtain morphology measures from galaxy image and add to collection.
        Picklable function for parallel loop
    """
    morph = get_morphology_measures(image)
    return morph


def get_morphology_measures_set_parallel(set: types.GeneratorType,
                                         measures_of_interest: list = measures_of_interest,
                                         N: int = 30):
    """ obtain morphology measures for a set of galaxy images
        parallel computation is about 6 times faster on 16 cores
    """
    if N > 0:
        set = [next(set) for i in range(N)]
    with Pool() as pool:
        morphs = pool.map(collect_morphology_measures, set)
    measures = {m: [] for m in measures_of_interest}
    for morph in morphs:
        for key, value in measures.items():
            value.append(getattr(morph, key))
    return measures


def get_morphology_measures_set(set: types.GeneratorType,
                                measures_of_interest: list = measures_of_interest,
                                N: int = 30,
                                ):
    """ obtain morphology measures for a set of galaxy images """
    measures = {m: [] for m in measures_of_interest}
    for i, image in enumerate(set):
        if i > N:
            break
        morph = get_morphology_measures(image)
        for key, value in measures.items():
            value.append(getattr(morph, key))
    return measures

# always use parallel version
get_morphology_measures_set = get_morphology_measures_set_parallel


# STATISTICAL INVESTIGATION --------------------------------



def plot_corner(*data, **kwargs):
    d = np.array(*data).T
    print(d.shape)
    corner(d, **kwargs)


def read_measures(keys: list, measures: dict):
    """ obtain group of measures from full dict

    Parameter
    ---------
    keys: list
        list of names (str) of measures to be read
    measures: dict
        full dict containing all measures
    """
    return [measures[k] for k in keys]


def read_measures_group(group: str, measures: dict):
    """ obtain group of measures from full dict

    Parameter
    ---------
    group: str
        name of group of morphology measures.
        One of "CAS", "MID", "gini-m20", "ellipticity"
        (keys of measures_groups)
    measures: dict
        full dict containing all measures
    """
    keys = measures_groups[group]
    return read_measures(keys, measures)


def plot_corner_measures_group(group: str, measures: dict, **kwargs):
    """ create corner plot of group of measures

    Parameter
    ---------
    group: str
        name of group of morphology measures.
        One of "CAS", "MID", "gini-m20", "ellipticity"
        (keys of measures_groups)
    measures: dict
        full dict containing all measures
    """
    labels = [m for m in measures_groups[group]]
    data = read_measures_group(group, measures)
    plot_corner(data, labels=labels, **kwargs)


def compute_distance_point_clouds_chamfer(
        points_source: torch.Tensor,
        points_target: torch.Tensor):
    """ compute the chamfer distance from source_points to target_points

    Parameter
    ---------
    source_points: torch.Tensor
        3D tensor of shape (N_batches, N_points, N_dimensions)
        contains points supposedly close to target points
    target_points: torch.Tensor
        3D tensor of shape (N_batches, N_points, N_dimensions)
        contains points from ground truth

    """
    chamfer_dist = ChamferDistance()
    dist = chamfer_dist(points_source.to(device), points_target.to(device))
    return dist.detach().cpu().item()


def transform_measures_to_points_chamfer(measures: dict):
    """ transform dict of measures to points needed to compute Chamfer distance

    Parameter
    ---------
    measures: dict
        contains M measures of interest (keys) for N samples (values)

    Output
    ------
    measures: torch.Tensor
        shape(1,N,M)

    """
    return torch.tensor([measures], requires_grad=False)


def compute_distance_measures_group(
        group: str,
        measures_source: dict,
        measures_target: dict,
        mode="chamfer", ):
    """ compute distance between points in group of measures

    Parameter
    ---------
    group: str
        name of group of morphology measures.
        One of "CAS", "MID", "gini-m20", "ellipticity"
        (keys of measures_groups)
    measures: dict
        full dict containing all measures
    """
    if mode == "chamfer":
        compute_distance_point_clouds = compute_distance_point_clouds_chamfer
        transform_measures = transform_measures_to_points_chamfer
    measures_source = read_measures_group(group, measures_source)
    measures_target = read_measures_group(group, measures_target)
    points_source = transform_measures(measures_source)
    points_target = transform_measures(measures_target)
    return compute_distance_point_clouds(points_source, points_target)

