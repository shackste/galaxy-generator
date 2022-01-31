# Here we put procedures for the statistical investigation and comparison
# of morphological properties of sets of galaxy images
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from corner import corner
from chamferdist import ChamferDistance

from . import measures


device = "cuda" if torch.cuda.is_available() else "cpu"



def plot_corner(*data, **kwargs):
    """ corner plot of data array (N_features, N_samples)

     Returns
     -------
     fig : matplotlib figure containing the corner plot
           to plot several datasets in on figure, pass kwarg fig=fig
     """
    d = np.array(*data).T
    print(d.shape)
    fig = corner(d, plot_contours=True, **kwargs)
    return fig

def plot_corner_measures_group(group: str, measures: measures.Measures, **kwargs):
    """ create corner plot of group of measures

    Parameter
    ---------
    group: str
        name of group of morphology measures.
        One of "CAS", "MID", "gini-m20", "ellipticity"
        (keys of measures.measures_groups)
    measures: dict
        full dict containing all measures

     Returns
     -------
     fig : matplotlib figure containing the corner plot
           to plot several datasets in on figure, pass kwarg fig=fig
    """
    data = measures.group(group)
    labels = data.keys
    fig = plot_corner(data.numpy(), labels=labels, **kwargs)
    return fig


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
    return dist.item()

compute_distance_point_clouds = compute_distance_point_clouds_chamfer


def compute_distance_measures_group(
        group: str,
        measures_source: measures.Measures,
        measures_target: measures.Measures):
    """ compute distance between points in group of measures

    Parameter
    ---------
    group: str
        name of group of morphology measures.
        One of "CAS", "MID", "gini-m20", "ellipticity"
        (keys of measures.measures_groups)
    measures: dict
        full dict containing all measures
    """
    return compute_distance_point_clouds(
        measures_source.group(group).torch(),
        measures_target.group(group).torch())

@torch.no_grad()
def evaluate_generator(dataloader, generator, plot=False, name=None, plot_path="~/Pictures/", test=False):
    """ evaluate galaxy image generator by computing the distance between
        morphology measures obtained from real images and
        morphology measures obtained from images generated from same labels.
        Distance is computed for point clouds for all groups in measures.measures_groups

    Parameters
    ----------
    dataloader: torch.DataLoader
            contains the target dataset
    generator: torch.Module
            contains the generator that transforms latent ant label vectors to galaxy images
    plot: boolean
            if True: plot corner plots for all groups
    name : identifier of the generator used in the filename and title

    Output
    ------
    distances: dict
            (chamfer) distance between point clouds of morphological measures for
            the real dataset and the generated counterparts
            for all groups in measures.measures_groups
    """
    # collect measures from dataset and generator
    target = measures.Measures()
    source = measures.Measures()
    for i, (images, labels) in tqdm(enumerate(dataloader), desc="get morphology measures"):
        measures_target = measures.get_morphology_measures_set(images)
        latent = torch.randn(len(images), generator.dim_z, device="cuda")
        labels = labels.cuda()
        images = generator(latent, labels)
        measures_generated = measures.get_morphology_measures_set(images)
        target += measures_target
        source += measures_generated

        if test and i > 10:
            break

    if test:
        for key, value in target.items():
            print("target", key, np.min(value), np.max(value))
        for key, value in source.items():
            print("source", key, np.min(value), np.max(value))

    # calculate distance between groups of point clouds
    distances = {}
    for group in measures.measures_groups.keys():
        distances[group] = compute_distance_measures_group(group, source, target)
        if plot:
            fig = None
            fig = plot_corner_measures_group(group, source, color="b", fig=fig, label_kwargs={"fontsize":16})
            fig = None
            fig = plot_corner_measures_group(group, target, fig=fig, color="r")
            fig.suptitle(name, fontsize=20)
            blue_line = mlines.Line2D([], [], color='blue', label='source')
            red_line = mlines.Line2D([], [], color='red', label='target')
            plt.legend(handles=[blue_line, red_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4, fontsize=16)
            plt.savefig(f"{plot_path}measures_{group}_{name}.png")
    distances["total"] = compute_distance_point_clouds(source.torch(), target.torch())
    return distances