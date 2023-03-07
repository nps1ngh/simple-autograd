"""
Layer Norm implementation.
Unlike PyTorch only normalizes over single feature *tensor* dimension.
"""
from typing import Optional, Union

import numpy as np

from ...variable import Variable

__all__ = ["layer_norm"]


# typedefs
Array = Union[Variable, np.ndarray]
Number = Union[int, float]
ArrayOrNumber = Union[Array, Number]


def layer_norm(
    input: Array,
    weight: Optional[Array] = None,
    bias: Optional[Array] = None,
    eps: float = 1e-05,
) -> Array:
    """
    Layer norm. Only normalizes last dimension, which is assumed to be the
    feature dimension. By dimension, we mean tensor's/array's dimensions.

    Parameters
    ----------
    input : Array
        The input to normalize. Shape (N, *, D)
    weight : Optional[Array]
        The weights for rescaling. Shape (D,)
    bias : Optional[Array]
        The bias. Shape (D,)
    eps : float
        The stabilizing term.

    Returns
    -------
    Array
        Normalized input.
    """
    result = input

    # calc stats
    mean = result.mean(axis=-1, keepdim=True)
    var = result.var(axis=-1, keepdim=True)

    # normalize
    result = result - mean
    if isinstance(var, Variable):
        denom = (var + eps).sqrt()
    else:
        denom = np.sqrt(var + eps)
    result = result / denom

    if weight is not None:
        result = result * weight
    if bias is not None:
        result = result + bias

    return result
