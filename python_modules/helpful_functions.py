""" helpful functions to investigate Network and performance
"""
import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite
import torch
from torch import rand
#from torchsummary import summary
from IPython.display import display
#from torchviz import make_dot

from file_system import folder_results
from parameter import image_dim, input_size


def show_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

#########################
## investigate network ##
#########################

def summarize(network: torch.nn.Module, input_size=input_size, graph=False, labels_dim=None) -> None:
    """ display summary of network with input of given size

    graph: display graph of network
    labels_dim: dimension of additional network input
    """
    net = network()
    net = net.cuda()
    print("input size", input_size)
    input_dummy = rand(3, *input_size).cuda()
    if labels_dim is not None:
        summary(net, input_size)
        labels_dummy = rand(labels_dim).cuda()
        y = net(input_dummy, labels_dummy)
    else:
        summary(net, input_size)
        y = net(input_dummy)
    if graph:
        display(make_dot(y, params=dict(list(net.named_parameters()))))

#########################
## investigate results ##
#########################

def write_RGB_image(*, image: np.array = None, filename : str = None) -> None:
    """ write RGB image to file """
    assert image is not None, "provide image"
    assert filename is not None, "provide filename"
    if not image.shape[2] == 3: ## pytorch: (colos,dim,dim), imwrite:(dim,dim,colors)
        image = np.rollaxis(image, 0, 3)
    imwrite(folder_results+filename, image)


def write_generated_galaxy_images_iteration(*, iteration: int = None, images: torch.Tensor = None, width: int = 8, height: int = 8) -> None:
    """ write set of galaxy images to file """
    assert type(iteration) is int, "provide iteration as integer"
    assert images is not None, "provide generated galaxy images"
    assert width*height == len(images), "width and height need to fit number of images, N = width*height"

    d = image_dim

    flat_image = np.empty((3, height*d, width*d))
    k = 0
    for iw in range(width):
        for ih in range(height):
            flat_image[:, ih*d:(ih+1)*d,iw*d:(iw+1)*d] = images[k]
            k += 1
    flat_image = (flat_image*255).astype("uint8")
    write_RGB_image(image=flat_image, filename=f"samples_iter{iteration}.png")

