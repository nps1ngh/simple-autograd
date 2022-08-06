"""
Contains activation functions as Modules.
"""
from .base import Module
from .. import functional as F


class ReLU(Module):
    def forward(self, input):
        return F.relu(input)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope)


