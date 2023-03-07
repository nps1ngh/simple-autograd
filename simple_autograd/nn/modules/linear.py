"""
A basic linear layer.
"""

import numpy as np

from ...variable import Variable
from .. import functional as F
from .base import Module


class Linear(Module):
    def __init__(self, in_dimensions: int, out_dimensions: int, bias: bool = True):
        super().__init__()

        self.in_dim = in_dimensions
        self.out_dim = out_dimensions

        # init like in torch.nn.Linear
        sqrt_k = 1 / np.sqrt(in_dimensions)
        self.weight = Variable(sqrt_k * np.random.rand(out_dimensions, in_dimensions))
        if bias:
            self.bias = Variable(sqrt_k * np.random.rand(out_dimensions))
        else:
            self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"bias={self.bias is not None}"
        )
