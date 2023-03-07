"""
Contains loss functions.
"""
from typing import Union

import numpy as np

from ...variable import Variable

__all__ = ["cross_entropy"]

# typedefs
Array = Union[Variable, np.ndarray]
Number = Union[int, float]
ArrayOrNumber = Union[Array, Number]


def cross_entropy(inputs: Array, targets: np.ndarray, reduction="mean"):
    assert inputs.ndim == 2
    assert targets.ndim == 1  # keep it simple
    assert targets.shape[0] == inputs.shape[0], "Batch dimensions should be the same!"

    batch_size = targets.shape[0]

    logsm = inputs.log_softmax(1)  # class dimension

    idx = (np.arange(batch_size), targets)
    losses = -logsm[idx]

    if reduction == "none":
        return losses
    elif reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        raise ValueError(f"Unknown reduction method specified: '{reduction}'")
