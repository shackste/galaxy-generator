""" helpful functions to investigate Network and performance
"""
import numpy as np
from imageio import imwrite
from torch import rand
from torchsummary import summary
from IPython.display import display
from torchviz import make_dot

from file_system import folder_results


def summarize(network, input_size=(3,64,64), graph=False, labels=False):
    net = network()
    net = net.cuda()
    print("input size", input_size)
    input_dummy = rand(3, *input_size).cuda()
    if labels:
        summary(net, input_size)
        labels_dummy = rand(labels_dim).cuda()
        y = net(input_dummy, labels_dummy)
    else:
        summary(net, input_size)
        y = net(input_dummy)
    if graph:
        display(make_dot(y, params=dict(list(net.named_parameters()))))


def write_generated_galaxy_image(*, image=None, filename=None):
    assert image is not None, "provide generated galaxy image"
    assert filename is not None, "provide filename"
    if not image.shape[2] == 3:
        image = np.rollaxis(image, 0, 3)
    imwrite(folder_results+filename, image)


def write_generated_galaxy_images_iteration(*, iteration=None, images=None):
    assert type(iteration) is int, "provide iteration as integer"
    assert images is not None, "provide generated galaxy images"

    w, h = 8, 8  ## !! find where these come from
    d = image_dim

    flat_image = np.empty((3, w*d, h*d))
    k = 0
    for iw in range(w):
        for ih in range(h):
            flat_image[:,iw*d:(iw+1)*d, ih*d:(ih+1)*d] = images[k]
            k += 1
    write_generated_galaxy_image(image=flat_image, filename=f"samples_iter{iteration}.png")
