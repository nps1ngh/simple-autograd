"""
Code is inspired by PyTorch's API.
"""
from .base import Module
from .batchnorm import BatchNorm2d
from .convolution import Conv2d
from .pooling import MaxPool2d, MinPool2d, AvgPool2d, GlobalAvgPool2d
from .linear import Linear
from .utils import (
    count_parameters,
    Identity,
    Sequential,
    Flatten,
    Exp,
    Sigmoid,
    Softmax,
)
from .act import ReLU, LeakyReLU
from .losses import CrossEntropyLoss
