""" Procedures for statistical tests of goodness of generated samples"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.ndimage.filters import gaussian_filter
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from cv2 import Laplacian

from photutils import detect_threshold, detect_sources
from statmorph import source_morphology


#########################################
#### multi-dimensional distributions ####
#########################################


################################################################################
## from https://stackoverflow.com/questions/22056667/kl-kullback-leibler-distance-with-histogram-smoothing-in-python
################################################################################

def kl(p: np.array, q: np.array) -> np.array:
    """ compute Kullback-Leibler distance between distributions p and q
    p and q are values of distribution functions with identical x
    usually renormalized histograms with same range and bins
    p and q can be multi-dimensional
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where((q != 0) * (p != 0), p * np.log(p / q), 0))


def smoothed_histogram_kl_distance(a: np.array, b: np.array, range=None, nbins=10, sigma=1) -> np.array:
    """ compute Kullback-Leibler distance between smoothed distribution of 1D data a and b
    """
    ahist, bhist = (np.histogram(a, range=range, bins=nbins, normed=True)[0],
                    np.histogram(b, range=range, bins=nbins, normed=True)[0])

    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))

    return kl(asmooth, bsmooth)


################################################################################


def smoothed_histogramdd_kl_distance(a: np.array, b: np.array, range=None, bins=10, sigma=1) -> np.array:
    """ compute Kullback-Leibler distance between smoothed distribution of data a and b
    a and b can have any dimension. for interpretation of dimensions, see docstring of np.histogramdd
    """
    assert range is not None, "please provide a range for all dimensions so the datasets can be compared properly"
    ahist, bhist = (np.histogramdd(a, range=range, bins=bins)[0],
                    np.histogramdd(b, range=range, bins=bins)[0])
    ## normalize to total number of data points to account for differences due to outliers
    ahist /= len(a)
    bhist /= len(b)


    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))

    return kl(asmooth, bsmooth)


def get_weighted_3Dprojected_mean(H: np.array, edges: list) -> np.array:
    """ compute weighed mean of 3D histogram, projected along third dimension """
    z_center = np.mean([edges[2][:-1], edges[2][1:]], axis=0)
    print(z_center)
    z_mean = np.resize(z_center, np.prod(H.shape)).reshape(*H.shape)

    norm = np.sum(H, axis=2)
    H_proj = np.sum(z_center*H, axis=2) * np.where(norm != 0, 1./norm, 0)
    return H_proj




def plot_projected_histogram3D(H: np.array, edges: list, axis_labels=None) -> None:
    """ plot projected map of 3D histogram.
    Third dimension is shown as colored average on 2D map

    """
    H_proj = get_weighted_3Dprojected_mean(H, edges)
    extent = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
    plt.imshow(H_proj.transpose()[::-1], extent=extent)
    if axis_labels:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
    plt.colorbar(label=axis_labels[2])


########################
#### total intesity ####
########################

def compute_total_intensities(image: torch.Tensor) -> list:
    """ return total intensity of individual color bands """
    segmap = get_segmentation_map(image)
    intensities = [np.sum(color[segmap]).item() for color in image]
    return intensities


def gather_total_intensity_data(images: torch.Tensor) -> np.array:
    """ return array containing total intensities of individual color bands in images """
    intensities = map(compute_total_intensities, images)
    return np.array(list(intensities))

def compute_average_intensity(images: torch.Tensor) -> float:
    intensities = gather_total_intensity_data(images)
    return np.mean(intensities)

def compute_divergence_total_intensity_statistics(images1: torch.Tensor, images2: torch.Tensor,
                                                  range=(1.5,3), bins=10) -> np.array:
    """ compute and return KL distance of distribution of log(total intensities)
    in different color bands of two sets of images
    """
    images1 = images1.numpy()
    images2 = images2.numpy()
    a, b = map(gather_total_intensity_data, [images1, images2])
    a, b = np.log10([a,b])
    range = [range] * len(images1[0]) ## same range for each colorband
    return smoothed_histogramdd_kl_distance(a, b, range=range, bins=bins)

#########################
#### image residuals ####
#########################

def compute_residual(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """ compute the residual of two images of same size """
    return ((image1 - image2)**2).sum().item()


def compute_residual_average_image(images1: torch.Tensor, images2: torch.Tensor) -> float:
    """ compute the residual of average of two sets of images, all of identical size """
    avg_img1 = images1.mean(axis=0)
    avg_img2 = images2.mean(axis=0)
    return compute_residual(avg_img1, avg_img2)


def get_residuals_randomly_picked_image_pairs(images: torch.Tensor, N=32) -> list:
    """ compute residuals of N pairs of prodived images """
    idx = list(np.random.choice(len(images), 2*N, replace=False))
    residuals = []
    while idx:
        image1, image2 = [images[idx.pop()] for i in range(2)]
        residuals.append(compute_residual(image1, image2))
    return residuals


def get_image_pair_residual_statistics(images: torch.Tensor, N=32) -> np.array:
    """ return average and standard deviation of residuals of N random pairs of images """
    residuals = get_residuals_randomly_picked_image_pairs(images, N=N)
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    return mu, sigma


###################
#### bluriness ####
###################


def compute_blurriness_metric(image_channel: torch.Tensor) -> float:
    """ compute bluriness metric as variance of laplacian  """
    lapl = Laplacian(image_channel.numpy(), 5)
    bluriness = lapl.var()
    return bluriness


grayscale_weights = torch.tensor([0.299, 0.587, 0.114]) ## RGB weights for eye-pleasing grayscale


def RGB2grayscale(image: torch.Tensor) -> torch.Tensor:
    scaled_image = torch.einsum("i,ijk->jk", grayscale_weights, image)
    return scaled_image


def gather_bluriness_metrics(images: torch.Tensor) -> list:
    """ collect bluriness metrics of provided RGB images after grayscaling them """
    bluriness = map(lambda image: compute_blurriness_metric(RGB2grayscale(image)), images)
    return list(bluriness)

def compute_average_bluriness(images: torch.Tensor) -> np.array:
    bluriness = gather_bluriness_metrics(images)
    return np.mean(bluriness)


def compute_divergence_bluriness_statistics(images1: torch.Tensor, images2: torch.Tensor,
                                            range=(-4,-2), bins=10) -> np.array:
    """ compute and return KL distance of distribution of bluriness metrics of grayscaled RGB images """
    b1, b2 = map(gather_bluriness_metrics, [images1, images2])
    b1, b2 = np.log10([b1,b2])
    return smoothed_histogramdd_kl_distance(b1, b2, range=(range,), bins=bins)




##################################
#### morphological statistics ####
##################################


def get_segmentation_map(image: torch.Tensor,
                         npixels=5, ## minimum number of connectied pixels
                         get_mask=False,
                         ) -> torch.Tensor:
    """ compute smoothed segmentation map marking the part of the image that contains the main source

    optional: when get_mask=True, also provide a mask map of all additional saurces
    """
    # get segmap from red colorband
    img = image[0]
    # create segmentation map
    threshold = detect_threshold(img, 1.5)
    segm = detect_sources(img, threshold, npixels)
    # Keep only the largest segment
    # ## !!! change this to take the central segment
    main_label = np.argmax(segm.areas) + 1
    segmap = segm.data == main_label
    # regularize shape
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5

    if not get_mask:
        return segmap
    else:
        # mask additional objects
        mask = np.zeros_like(img).astype("bool")
        for obj_label in range(1,len(segm.areas)+1):
            if obj_label == main_label:
                continue
            mask[segm.data == obj_label] = True
        return segmap, mask

def remove_background(image: torch.Tensor,
                      box_size=4, filter_size=3,
                      mask=None, pass_background=False) -> torch.Tensor:
    """ remove background noise from given image """
    sigma_clip = SigmaClip(sigma=3.)
    #    bkg_estimator = MedianBackground()
    bkg_estimator = SExtractorBackground()

    bkg = Background2D(image, box_size=box_size, filter_size=filter_size,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                       mask=mask,
                       )
    clean_image = image - bkg.background
    return (clean_image, bkp.background) if pass_background else clean_image


def extract_morphological_properties(image: torch.Tensor,
                              gain=1e4, ## assumed average electron counts per pixel at effective radius
                              npixels=5, ## minimum number of connectied pixels
                              plot=False) -> list:
    """ extract non-parametric morphological diagnostics from galaxy image using https://github.com/vrodgom/statmorph
    image is 2D representation of galaxy, either one colorband or total intensity (!!! check which is best, red-band?)

    !!! image has to be background-subtracted !!!
    """
    segmap, mask = get_segmentation_map(image, npixels=npixels, get_mask=True)
    clean_image, background = remove_background(image, box_size=4, filter_size=3, mask=mask + segmap, pass_background=True) ## mask all sources

    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs[0][0].set_title("original image")
        img = axs[0][0].imshow(image)
        fig.colorbar(img, ax=axs[0][0])
        axs[0][1].set_title("segmentation map")
        axs[0][1].imshow(segmap, cmap='gray')
        axs[0][2].set_title("mask")
        axs[0][2].imshow(~mask, cmap='gray')
        axs[1][0].set_title("backgroung-subtracted image")
        img = axs[1][0].imshow(clean_image)
        fig.colorbar(img, ax=axs[1][0])
        axs[1][1].set_title("background noise")
        img = axs[1][1].imshow(background)
        fig.colorbar(img, ax=axs[1][1])

        axs[1][2].set_title("segmentation data")
        img = axs[1][2].imshow(segm.data)
        fig.colorbar(img, ax=axs[1][2])

        plt.show()
    source_morphs = source_morphology(clean_image, segmap, gain=gain, mask=mask)
    return source_morphs[0]
