"""
Convolution layer
"""
from typing import Union

import numpy as np

from .base import Module
from .. import functional as F
from ...variable import Variable


def _ensure_tuple(x: Union[int, tuple[int, int]]):
    if isinstance(x, int):
        x = (x, x)
    assert isinstance(x, tuple), f"Tuple or int should be passed, but was {type(x)}"
    return x


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple[int, int]], bias: bool = True,
                 padding: Union[int, tuple[int, int]] = (0, 0)):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.kernel_size = _ensure_tuple(kernel_size)
        self.padding = _ensure_tuple(padding)

        # init like in torch.nn.Linear
        sqrt_k = 1 / np.sqrt(in_channels)
        self.weight = Variable(sqrt_k * np.random.rand(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Variable(sqrt_k * np.random.rand(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        result = F.conv2d(x, self.weight, padding=self.padding)
        if self.bias is not None:
            result = result + self.bias

    def extra_repr(self) -> str:
        return (
            f"in_ch={self.in_ch}, "
            f"out_ch={self.out_ch}, "
            f"bias={self.bias is not None}"
        )
