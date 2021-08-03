""" neural network sequetial layer operations not included in pytorch
"""

from torch.nn import Module

class Reshape(Module):
    """ reshape tensor to given shape

        USAGE
        -----
        >>> torch.nn.Sequential(
        >>>    ...
        >>>    Reshape(*shape)
        >>>    ...
        >>> )
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class PrintShape(Module):
    """ echo shape of tensor, use for debugging

        USAGE
        -----
        >>> torch.nn.Sequential(
        >>>     ...
        >>>     PrintShape()
        >>>     ...
        >>> )
    """
    def __init__(self):
        super(PrintShape, self).__init__()
    def forward(self, input):
        print(input.shape)
        return input
