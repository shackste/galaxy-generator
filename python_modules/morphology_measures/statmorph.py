# here we implement the use of statmorph package
# https://github.com/vrodgom/statmorph
# to obtain morphological measures from images of galaxies,
# focussing on cropped images of the Galaxy Zoo dataset
# https://kaggle.com/c/galaxy-zoo-the-galaxy-challenge/

from contextlib import ExitStack, redirect_stderr, redirect_stdout

import numpy as np
import scipy.ndimage as ndi
import photutils
import statmorph
import torch
from skimage.color import rgb2gray




def get_morphology_measures(image: torch.Tensor,
                            gain: float = 10000.0, # assume average of 10,000 electrons per pixel
                            silent: bool = True
                           ):
    """ return the morphology measures of a galaxy in an RGB image """
    image = image[0].transpose(0,2)
    image = image.numpy()
    image_bw = RGB2BW(image)
    segmap = get_segmentation_map(image_bw)
    with ExitStack() as stack:
        if silent:
            stack.enter_context(redirect_stdout(open(os.devnull, 'w')))
            stack.enter_context(redirect_stderr(open(os.devnull, 'w')))
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
