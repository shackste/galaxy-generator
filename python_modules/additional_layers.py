""" neural network sequetial layer operations not included in pytorch
"""

import torch
from torch.nn import Module, Sequential, Linear, MaxPool1d
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ALReLU(Module):
    """ Leaky Relu with absolute negative part
        https://arxiv.org/pdf/2012.07564v1.pdf
    """
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super(ALReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        x = torch.abs(torch.nn.functional.leaky_relu(x, self.negative_slope, self.inplace))
        return x

class MaxOut(Module):
    """ Maxout Layer: take max value from N_layers Linear layers
    take MaxPool1d of of single  Linear layer with N_layers*out_features nodes
    """
    def __init__(self, in_features: int, out_features: int, N_layers: int = 2, **kwargs):
        super(MaxOut, self).__init__()
        self.maxout = Sequential(
            Linear(in_features, N_layers*out_features, **kwargs),
            Reshape(1,N_layers*out_features),
            MaxPool1d(N_layers),
            Reshape(out_features),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxout(x)
        return x


class Conv2dUntiedBias(Module):
    """
    2D Convolutional Layer with untied bias, i. e. individual trainable bias for each output pixel and feature
    
    Parameter
    ---------
    height(int) : height of output pixel map
    width(int) : width of output pixel map
    in_channels(int) : number of input channels
    out_channels(int) : number of output channels
    bias_init(float) : initial value for biases
    weight_std(float) : normal standard deviation for initial weights
    """
    def __init__(self, height: int, width: int, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups=1, bias_init=0.1, weight_std=0.01):
        super(Conv2dUntiedBias, self).__init__() 
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels, height, width))
        self.reset_parameters(bias_init=bias_init, weight_std=weight_std)

    def reset_parameters(self, bias_init=None, weight_std=None) -> None:
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        if weight_std is None:
            stdv = n**-0.5
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weight.data.uniform_(-weight_std, weight_std)            
        if bias_init is None:
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias.data[:] = bias_init
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.kernel_size[0] % 2 and self.padding: ## one-sided padding 1 extra for even kernel size
            x = F.pad(input=x, pad=(1,0,1,0))
        output = F.conv2d(x, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
        # add untied bias
        output += self.bias.unsqueeze(0) #.repeat(input.size(0), 1, 1, 1)
        return output


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


