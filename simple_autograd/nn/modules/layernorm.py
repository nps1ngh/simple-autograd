"""
Layer norm.
"""
from typing import Optional

import numpy as np

from ...variable import Variable
from .. import functional as F
from .base import Module


class LayerNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-05, affine: bool = True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        self.weight: Optional[Variable] = None
        self.bias: Optional[Variable] = None
        if self.affine:
            self.weight = Variable(np.empty(num_features))
            self.bias = Variable(np.empty(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.view(np.ndarray).fill(1.0)
            self.bias.view(np.ndarray).fill(0.0)

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, affine={self.affine}"

    def forward(self, x):
        return F.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
