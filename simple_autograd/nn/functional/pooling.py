"""
Pooling operations with grad support.

Requires skimage!
"""
from typing import Union, Tuple

import numpy as np

from ...variable import Variable

__all__ = ["max_pool2d", "min_pool2d", "avg_pool2d"]

# typedefs
Array = Union[Variable, np.ndarray]
Number = Union[int, float]
ArrayOrNumber = Union[Array, Number]


def _get_2D_patches(
    input: "Array", patch_size: "Tuple[int, int]", flatten: "bool" = False,
) -> "Array":
    """
    Create 2D patches of given patch_size out of given input.
    Input is "shortened" to a multiple of patch_size in each of
    the H and W dimensions.

    Parameters
    ----------
    input : Array
        The input to extract patches from.
    patch_size : Tuple[int, int]
        The size of the patches to extract.
    flatten : bool
        Whether to convert the 2D patches to 1D.

    Returns
    -------
    Array
        The extracted patches.
    """
    N, C, H, W = input.shape
    kH, kW = patch_size

    # "shorten" array
    input = input[..., :(H // kH) * kH, :(W // kW) * kW]

    # from https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array
    patches = input.reshape((N, C, H // kH, kH, W // kW, kW)).swapaxes(3, 4)
    if flatten:
        # flatten the 2D patches to 1D
        patches = patches.reshape((N, C, H // kH, W // kW, -1))
    return patches


def _ensure_2d_tuple(k):
    if isinstance(k, int):
        k = (k, k)
    return k


def max_pool2d(input, kernel_size):
    """
    Does max pooling on 2D input.

    Parameters
    ----------
    input : Array
        The 2D input. Shape [N, C, H, W]
    kernel_size : int | Tuple[int, int]
        The size of the kernel/window.

    Returns
    -------
    Array
        Max pooling result.
    """
    assert input.ndim == 4, f"4D input expected! Got {input.ndim}D!"
    kernel_size = _ensure_2d_tuple(kernel_size)

    patches = _get_2D_patches(input, kernel_size, flatten=True)
    return patches.max(-1)


def min_pool2d(input, kernel_size):
    """
    Does min pooling on 2D input.

    Parameters
    ----------
    input : Array
        The 2D input. Shape [N, C, H, W]
    kernel_size : int | Tuple[int, int]
        The size of the kernel/window.

    Returns
    -------
    Array
        Min pooling result.
    """
    assert input.ndim == 4, f"4D input expected! Got {input.ndim}D!"
    kernel_size = _ensure_2d_tuple(kernel_size)

    patches = _get_2D_patches(input, kernel_size, flatten=True)
    return patches.min(-1)


def avg_pool2d(input, kernel_size):
    """
    Does avg pooling on 2D input.

    Parameters
    ----------
    input : Array
        The 2D input. Shape [N, C, H, W]
    kernel_size : int | Tuple[int, int]
        The size of the kernel/window.

    Returns
    -------
    Array
        Avg pooling result.
    """
    assert input.ndim == 4, f"4D input expected! Got {input.ndim}D!"
    kernel_size = _ensure_2d_tuple(kernel_size)

    patches = _get_2D_patches(input, kernel_size, flatten=True)
    return patches.mean(-1)



