"""
Batch norm 2d layer.
"""
from typing import Optional

import numpy as np

from ...variable import Variable
from .. import functional as F
from .base import Module


class _BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.weight: Optional[Variable] = None
        self.bias: Optional[Variable] = None
        if self.affine:
            self.weight = Variable(np.empty(num_features))
            self.bias = Variable(np.empty(num_features))

        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None
        self.num_batches_tracked: Optional[np.ndarray] = None
        if self.track_running_stats:
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            self.num_batches_tracked = np.zeros(1, dtype=np.longlong)  # int64

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.fill(0.0)
            self.running_var.fill(1.0)
            self.num_batches_tracked.fill(0)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.view(np.ndarray).fill(1.0)
            self.bias.view(np.ndarray).fill(0.0)

    def extra_repr(self):
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(_BatchNorm):
    def forward(self, input):
        """
        This is taken from
        https://github.com/pytorch/pytorch/blob/v1.11.0/torch/nn/modules/batchnorm.py
        """
        assert input.ndim == 2, f"Expected 2D input, got {input.ndim}D instead!"

        momentum = self.momentum
        if (
            self.training
            and self.track_running_stats
            and self.num_batches_tracked is not None
        ):
            self.num_batches_tracked += 1
            if momentum is None:
                # cumulative
                momentum = 1.0 / float(self.num_batches_tracked)

        if not self.training or self.track_running_stats:
            r_mean = self.running_mean
            r_var = self.running_var
        else:
            r_mean = None
            r_var = None

        if self.training:
            bn_training = True
        else:
            bn_training = (r_mean is not None) and (r_var is not None)

        return F.batch_norm_1d(
            input,
            r_mean,
            r_var,
            weight=self.weight,
            bias=self.bias,
            training=bn_training,
            momentum=momentum,
            eps=self.eps,
        )


class BatchNorm2d(_BatchNorm):
    def forward(self, input):
        """
        This is taken from
        https://github.com/pytorch/pytorch/blob/v1.11.0/torch/nn/modules/batchnorm.py
        """
        assert input.ndim == 4, f"Expected 4D input, got {input.ndim}D instead!"

        momentum = self.momentum
        if (
            self.training
            and self.track_running_stats
            and self.num_batches_tracked is not None
        ):
            self.num_batches_tracked += 1
            if momentum is None:
                # cumulative
                momentum = 1.0 / float(self.num_batches_tracked)

        if not self.training or self.track_running_stats:
            r_mean = self.running_mean
            r_var = self.running_var
        else:
            r_mean = None
            r_var = None

        if self.training:
            bn_training = True
        else:
            bn_training = (r_mean is not None) and (r_var is not None)

        return F.batch_norm_2d(
            input,
            r_mean,
            r_var,
            weight=self.weight,
            bias=self.bias,
            training=bn_training,
            momentum=momentum,
            eps=self.eps,
        )
