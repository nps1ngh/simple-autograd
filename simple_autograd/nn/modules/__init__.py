"""
Code is inspired by PyTorch's API.
"""
from .act import LeakyReLU, ReLU
from .attention import MultiHeadSelfAttention
from .base import Module
from .batchnorm import BatchNorm1d, BatchNorm2d
from .convolution import Conv2d
from .layernorm import LayerNorm
from .linear import Linear
from .losses import CrossEntropyLoss
from .pooling import AvgPool2d, GlobalAvgPool2d, MaxPool2d, MinPool2d
from .utils import (
    Exp,
    Flatten,
    Identity,
    Sequential,
    Sigmoid,
    Softmax,
    count_parameters,
)
