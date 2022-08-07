"""
Convolution2d implementation.

This one is special and was perhaps the most challenging one.

The additional dependency on scipy comes from here
"""
from typing import Union

import numpy as np
from scipy import signal

from ...operations import BinaryOperator
from ...variable import Variable

__all__ = ["conv2d"]

# typedefs
Array = Union[Variable, np.ndarray]
Number = Union[int, float]
ArrayOrNumber = Union[Array, Number]


def _conv2d(input: np.ndarray, weight: np.ndarray, padding: Union[int, tuple[int, int]] = 0) -> np.ndarray:
    """
    The actual implementation. This does not do any graph attachment etc.
    For that use `conv2d` instead.
    Note that this implementation is of course much slower than compared to torch's.

    :param input: The input to convolve over. [N, C_in, H, W]
    :type input: np.ndarray
    :param weight: The kernel weights. [C_out, C_in, K_h, K_w]
    :type weight: np.ndarray
    :param padding: The padding to use. If int then pads both h and w with.
        Otherwise, a tuple can be used to specify separate paddings for h and w.
    :type padding: int | tuple[int, int]
    :return: The convolution 2d result
    :rtype: np.ndarray [N, C_out, R_h, R_w]
    """
    assert input.ndim == 4
    assert weight.ndim == 4
    assert (
            weight.shape[1] == input.shape[1]
    ), f"expected same number of input channels! {weight.shape=} and {input.shape=}"

    if isinstance(padding, int):
        h_padding = w_padding = (padding, padding)
    elif isinstance(padding, tuple):
        assert len(padding) == 2, f"a pair expected but got a {len(padding)}-tuple!"
        assert padding[0] > 0
        assert padding[1] > 0
        h_padding = padding[:1] * 2
        w_padding = padding[1:] * 2
    else:
        raise TypeError(f"padding should be an int or a tuple of ints! Given: {padding}")

    # this is the simplest way of doing it
    input = np.pad(input, [(0, 0), (0, 0), h_padding, w_padding])  # only pad h and w
    del padding
    del h_padding
    del w_padding

    # determine output size and init output array
    batch_size, in_channels, h, w = input.shape
    out_channels, _, h_k, w_k = weight.shape
    h_result = 1 + (h - h_k)
    w_result = 1 + (w - w_k)
    output = np.empty((batch_size, out_channels, h_result, w_result))

    for b in range(batch_size):  # select image
        for k in range(out_channels):  # select kernel
            output[b, k] = signal.correlate(input[b], weight[k], mode="valid")

    return output


def _conv2d_backward_wrt_input(weight: np.ndarray, out_grad: np.ndarray, input_shape: tuple,
                               padding: tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Calculates the gradient with respect to the input.
    In particular, the larger *batched* operand.

    :param weight: The weight as a np.ndarray (not variable). [C_out, C_in, K_h, K_w]
    :type weight: np.ndarray
    :param out_grad: The incoming gradient of the output. [N, C_out, R_h, R_w]
    :type out_grad: np.ndarray
    :param input_shape: The shape of the original input
    :type input_shape: tuple
    :param padding: The padding for the input used.
    :type padding: tuple[int, int]
    :return: The gradient for the input operand. [N, C_in, H, W]
    :rtype: np.ndarray
    """
    c_out, _, k_h, k_w = weight.shape
    batch_size, c_in, h, w = input_shape
    p_h, p_w = padding

    # get w padded
    p_k_h = h - k_h + p_h
    p_k_w = w - k_w + p_w
    weight = np.pad(weight, [(0, 0), (0, 0), (p_k_h, p_k_h), (p_k_w, p_k_w)])

    # init
    grad = np.empty(input_shape)

    for b in range(batch_size):  # select image
        for c in range(c_in):  # select input channel
            grad[b, c] = signal.correlate(weight[:, c], out_grad[b], mode="valid")

    return grad


def _conv2d_backward_wrt_weight(input: np.ndarray, out_grad: np.ndarray, weight_shape: tuple,
                               padding: tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Calculates the gradient with respect to the weight.
    In particular, the smaller un-batched operand.

    :param input: The input as a np.ndarray (not variable). [N, C_in, H, W]
    :type input: np.ndarray
    :param out_grad: The incoming gradient of the output. [N, C_out, R_h, R_w]
    :type out_grad: np.ndarray
    :param weight_shape: The shape of the weight
    :type weight_shape: tuple
    :param padding: The padding for the input used.
    :type padding: tuple[int, int]
    :return: The gradient for the input operand. [C_out, C_in, K_h, K_w]
    :rtype: np.ndarray
    """
    batch_size, c_in, h, w = input.shape
    c_out = weight_shape[0]
    p_h, p_w = padding

    # get input padded
    input = np.pad(input, [(0, 0), (0, 0), (p_h, p_h), (p_w, p_w)])

    # init
    grad = np.empty(weight_shape)

    for c in range(c_in):  # select channel
        for k in range(c_out):  # select kernel
            grad[k, c] = signal.correlate(input[:, c], out_grad[:, k], mode="valid")

    return grad


class Conv2dBackward(BinaryOperator):
    def __init__(self, input: Variable, weight: Variable, padding: Union[int, tuple[int, int]] = 0):
        # a -> input image
        # b -> kernel
        super().__init__(a=input, b=weight)

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

    def backprop(self, out_grad: np.ndarray) -> None:
        # note that a is much larger than b
        # => pad b when gradient for a is required
        if self.a.requires_grad:
            a_grad = _conv2d_backward_wrt_input(self.b.view(np.ndarray), out_grad, self.a.shape, self.padding)
            self._update_grad(self.a, a_grad)

        # simpler case
        if self.b.requires_grad:
            b_grad = _conv2d_backward_wrt_weight(self.a.view(np.ndarray), out_grad, self.b.shape, self.padding)
            self._update_grad(self.b, b_grad)


def conv2d(input: Array, weight: Array, padding: Union[int, tuple[int, int]] = 0) -> Array:
    """
    Calculates the convolution of input with the weight kernel.

    :param input: The input to convolve over. [N, C_in, H, W]
    :type input: np.ndarray
    :param weight: The kernel weights. [C_out, C_in, K_h, K_w]
    :type weight: np.ndarray
    :param padding: The padding to use. If int then pads both h and w with.
        Otherwise, a tuple can be used to specify separate paddings for h and w.
    :type padding: int | tuple[int, int]
    :return: The convolution 2d result
    :rtype: np.ndarray [N, C_out, R_h, R_w]
    """
    result = _conv2d(input.view(np.ndarray), weight.view(np.ndarray), padding=padding)

    if isinstance(input, Variable) or isinstance(weight, Variable):
        input = Variable.ensure_is_variable(input)
        weight = Variable.ensure_is_variable(weight)

        requires_grad = input.requires_grad or weight.requires_grad,
        grad_fn = {
            "grad_fn":Conv2dBackward(input, weight, padding=padding)
        } if requires_grad else {}
        result = Variable(
            result,
            requires_grad=requires_grad,
            retains_grad=False,
            **grad_fn,
        )

    return result
