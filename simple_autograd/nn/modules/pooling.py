"""
Pooling layers
"""
from .. import functional as F
from .base import Module


class _Pool2d(Module):
    def __init__(self, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, tuple):
            raise ValueError("kernel size should be either an int or tuple!")

        self.kernel_size = kernel_size


class MaxPool2d(_Pool2d):
    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size)


class MinPool2d(_Pool2d):
    def forward(self, x):
        return F.min_pool2d(x, self.kernel_size)


class AvgPool2d(_Pool2d):
    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size)


class GlobalAvgPool2d(Module):
    def forward(self, x):
        assert x.ndim == 4, f"Expected 4D input, but got {x.ndim}D!"
        return x.mean((2, 3))
