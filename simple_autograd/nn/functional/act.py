"""
Contains activation functions.
"""
from typing import Union

import numpy as np

from ...variable import Variable

__all__ = ["relu", "leaky_relu"]

# typedefs
Array = Union[Variable, np.ndarray]
Number = Union[int, float]
ArrayOrNumber = Union[Array, Number]


def relu(input: ArrayOrNumber):
    if isinstance(input, Variable):
        return input.relu()
    else:
        return np.maximum(input, 0)


def leaky_relu(input: ArrayOrNumber, negative_slope: Number = 0.01):
    if isinstance(input, Variable):
        return input.maximum(0) + negative_slope * input.minimum(0)
    else:
        return np.maximum(input, 0)
