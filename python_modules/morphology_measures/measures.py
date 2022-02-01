# here we implement the use of statmorph package
# https://github.com/vrodgom/statmorph
# to obtain morphological measures from images of galaxies,
# focussing on cropped images of the Galaxy Zoo dataset
# https://kaggle.com/c/galaxy-zoo-the-galaxy-challenge/

import types
from io import StringIO
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from multiprocessing import Pool
from collections.abc import Iterable

import numpy as np
import scipy.ndimage as ndi
import photutils
import statmorph
import torch
from skimage.color import rgb2gray

measures_groups = {"CAS": ["concentration", "asymmetry", "smoothness", ],
                   "MID": ["multimode", "intensity", "deviation", ],
                   "gini-m20": ["m20", "gini", ],
                   "ellipticity": ["ellipticity_asymmetry", ], }


measures_of_interest = [m
                        for measures in measures_groups.values()
                        for m in measures]

measures_groups["all"] = measures_of_interest


def get_morphology_measures(image: torch.Tensor,
                            gain: float = 10000.0, # assume average of 10,000 electrons per pixel
                            silent: bool = True
                           ):
    """ return the morphology measures of a galaxy in an RGB image """
    if len(image.shape) == 4:
        image = image[0]
    image = image.transpose(0,2)
    image = image.detach().cpu().numpy()
    image_bw = RGB2BW(image)
    segmap = get_segmentation_map(image_bw)
    with ExitStack() as stack:
        if silent:
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(redirect_stderr(StringIO()))
        morphology_measures = statmorph.source_morphology(image_bw, segmap, gain=gain)
    return morphology_measures[0]

def get_segmentation_map(image: np.array, npixels: int = 5):
    """ obtain segmentation map of biggest object in image

    Parameter
    ---------
    image : numpy.array
        contains the b/w image in 2D
    npixers : int
        minimum number of connected pixels
    """
    threshold = photutils.detect_threshold(image, 1.5)
    segm = photutils.detect_sources(image, threshold, npixels)
    # Keep only the largest segment
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    # regularize
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5
    return segmap


def RGB2BW(image):
    """ Transform RGB image to BW image """
    return rgb2gray(image)


class Measures:
    def __init__(self, measures=measures_of_interest):
        """ Initialize a Measures object.

        Parameters
        ----------
        measures : list of strings
                names of measures contained in the object.
        """
        self.keys = measures
        for measure in self.keys:
            setattr(self, measure, [])

    def __add__(self, other):
        for key, value in self.items():
            addition = getattr(other, key)
            if isinstance(addition, Iterable):
                value.extend(addition)
            else:
                value.append(addition)
        return self

    def clean_measures(self, clean: dict = {}):
        """ remove datapoints that exceed the limits defined in clean

        Parameters
        ----------
        clean : dict, e. g. {"x":(0,1)}
                lists minimum and maximum range for certain measures.
                data points exceeding this limit will be dropped.
         """
        remove = []
        outliers = []
        for key, value in self.items():
            if key in clean:
                mini, maxi = clean[key]
                outliers.extend([id for id, x in enumerate(value) if x < mini or x > maxi])
        outliers = sorted(list(set(outliers)), reverse=True)
        for key, value in self.items():
            for o in outliers:
                del value[o]
        print(f"{len(outliers)} outliers removed.")

    def items(self):
        """ mimic behavior of dict"""
        for key in self.keys:
            yield key, getattr(self, key)

    def measures(self, keys):
        """ return a subset of Measures """
        measures = Measures(measures=keys)
        measures.__add__(self)
        return measures

    def group(self, group):
        """ return certain group of measures """
        keys = measures_groups[group]
        return self.measures(keys)

    def numpy(self):
        """ return measures as numpy array """
        data = [value for key, value in self.items()]
        return np.array(data)

    def torch(self):
        """ return measures as torch tensor

         Output
         ------
         measures: torch.Tensor
                tensor of shape (1,N,M) with N points of M measures
         """
        return torch.from_numpy(self.numpy().T)[None,...]


def get_morphology_measures_set_parallel(set: types.GeneratorType,
                                         measures_of_interest: list = measures_of_interest,
                                         N: int = 0):
    """ obtain morphology measures for a set of galaxy images
        parallel computation is about 6 times faster on 16 cores
    """
    if N > 0:
        set = [next(set) for i in range(N)]
    with Pool() as pool:
        morphs = pool.map(get_morphology_measures, set)
#    morphs = map(get_morphology_measures, set)
    measures = Measures()
    for morph in morphs:
        measures += morph
    return measures


def get_morphology_measures_set(set: types.GeneratorType,
                                measures_of_interest: list = measures_of_interest,
                                ):
    """ obtain morphology measures for a set of galaxy images """
    measures = Measures(keys=measures_of_interest)
    for image in set:
        measures += get_morphology_measures(image)
    return measures

# always use parallel version
get_morphology_measures_set = get_morphology_measures_set_parallel
