import torch.nn as nn

from python_modules.evaluation.utils import get_feature_detector


def vgg16() -> nn.Module:
    """Returns VGG16 with LPIPS, used in StyleGAN2-ADA

    Returns:
        nn.Module: pretrained VGG16 network
    """

    # VGG16, used in the StyleGAN
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'

    model = get_feature_detector(url)
    return model
