"""
Contains layers as functions.
"""
from typing import Optional, Union

import numpy as np

from ...variable import Variable

__all__ = ["linear"]

# typedefs
Array = Union[Variable, np.ndarray]
Number = Union[int, float]
ArrayOrNumber = Union[Array, Number]


def linear(input: Array, weight: Array, bias: Optional[ArrayOrNumber] = None) -> Array:
    """
    Calculates input @ weight^T + bias.

    In other words,

    $$ xA^\intercal + b$$
    """
    if isinstance(weight, Variable):
        w = weight.t()
    else:
        weight = np.atleast_2d(weight)
        w = np.swapaxes(weight, 0, 1)

    result = input @ w

    if bias is not None:
        result = result + bias

    return result
