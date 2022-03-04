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
from . import wasserstein


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

def compute_distance_point_clouds_wasserstein(
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
    print("shape", points_source.shape, points_target.shape)
    dist = wasserstein.wasserstein(points_source.to(device), points_target.to(device))
    return dist.item()

compute_distance_point_clouds = compute_distance_point_clouds_wasserstein


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

def get_measures_dataloader(dataloader) -> measures.Measures:
    """ get morphology measures for images in dataloader """
    data = measures.Measures()
    for images, _ in dataloader:
        data += measures.get_morphology_measures_set(images)
    return data

@torch.no_grad()
def get_measures_generator(generator, dataloader) -> measures.Measures:
    data = measures.Measures()
    for _, labels in tqdm(dataloader, desc="get morphology measures"):
        latent = torch.randn(labels.shape[0], generator.dim_z, device="cuda")
        labels = labels.cuda()
        images = generator(latent, labels)
        data += measures.get_morphology_measures_set(images.cpu())
    return data


def evaluate_measures(target: measures.Measures, data: measures.Measures, plot=False, name=None, plot_path="~/Pictures/") -> dict:
    """ calculate distance between groups of point clouds """
    distances = {}
    for group in measures.measures_groups.keys():
        distances[group] = compute_distance_measures_group(group, data, target)
        if plot and not group == "ellipticity": # ellipticity is a single measure, corner plot is useless
            fig = None
            fig = plot_corner_measures_group(group, data, color="b", fig=fig, label_kwargs={"fontsize":16})
            fig = plot_corner_measures_group(group, target, fig=fig, color="r")
            fig.suptitle(name, fontsize=20)
            blue_line = mlines.Line2D([], [], color='blue', label='source')
            red_line = mlines.Line2D([], [], color='red', label='target')
            plt.legend(handles=[blue_line, red_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4, fontsize=16)
            plt.savefig(f"{plot_path}measures_{group}_{name}.png")
    distances["total"] = compute_distance_point_clouds(data.torch(), target.torch())
    return distances


@torch.no_grad()
def evaluate_generator(dataloader, generator, plot=False, name=None, plot_path="~/Pictures/", test=False, target_measures: measures.Measures = None, clean: dict = {}):
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
    test : boolean
            if True: break after one batch for code testing
    target_measures : measures.Measures
            contains measures_target obtained in previous runs
    clean : dict, e. g. {"deviation":(0,1)}
            lists minimum and maximum range for certain measures.
            data points exceeding this limit will be dropped.

    Output
    ------
    distances: dict
            (chamfer) distance between point clouds of morphological measures for
            the real dataset and the generated counterparts
            for all groups in measures.measures_groups
    measures_target : np.array
            contains the measures for target which can be passed to kwarg target_measures to save computiation time in future runs
    """
    # collect measures from dataset and generator
    target = measures.Measures()
    source = measures.Measures()
    for i, (images, labels) in tqdm(enumerate(dataloader), desc="get morphology measures"):
        if not target_measures:
            measures_target = measures.get_morphology_measures_set(images)
            target += measures_target
        latent = torch.randn(len(images), generator.dim_z, device="cuda")
        labels = labels.cuda()
        images = generator(latent, labels)
        measures_generated = measures.get_morphology_measures_set(images)
        source += measures_generated

        if test: # and i > 2:
            break

    if test:
        for key, value in target.items():
            print("target", key, np.min(value), np.max(value))
        for key, value in source.items():
            print("source", key, np.min(value), np.max(value))

    if target_measures:
        target = target_measures.copy()
    if clean:
        source.clean_measures(clean)
        target.clean_measures(clean)

    # calculate distance between groups of point clouds
    distances = {}
    for group in measures.measures_groups.keys():
        distances[group] = compute_distance_measures_group(group, source, target)
        if plot:
            print(group)
            if group == "ellipticity": # ellipticity is a single measure, corner plot is useless
                continue
            fig = None
            fig = plot_corner_measures_group(group, source, color="b", fig=fig, label_kwargs={"fontsize":16})
            fig = plot_corner_measures_group(group, target, fig=fig, color="r")
            fig.suptitle(name, fontsize=20)
            blue_line = mlines.Line2D([], [], color='blue', label='source')
            red_line = mlines.Line2D([], [], color='red', label='target')
            plt.legend(handles=[blue_line, red_line], bbox_to_anchor=(0., 1.0, 1., .0), loc=4, fontsize=16)
            plt.savefig(f"{plot_path}measures_{group}_{name}.png")
    distances["total"] = compute_distance_point_clouds(source.torch(), target.torch())
    return distances, measures_target
