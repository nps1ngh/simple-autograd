"""
Batch Norm implementation.
"""
from typing import Union, Optional

import numpy as np

from ...operations import Operator
from ...variable import Variable


__all__ = ["batch_norm_1d", "batch_norm_2d"]


# typedefs
# note: Only np.ndarray => no Variable
Array = Union[Variable, np.ndarray]
Number = Union[int, float]
ArrayOrNumber = Union[Array, Number]


# overwrite simple implementation with a more efficient one
MORE_EFFICIENT = False


def batch_norm_1d(
        input: Array,
        running_mean: Optional[np.ndarray] = None,
        running_var: Optional[np.ndarray] = None,
        weight: Optional[Array] = None,
        bias: Optional[Array] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-05,
) -> Array:
    assert input.ndim == 2

    result = input
    if training:
        mean = result.mean(axis=0)
        var = result.var(axis=0)

        # update stats if given
        if running_mean is not None:
            running_mean[:] = (1 - momentum) * running_mean + np.multiply(momentum, mean.data)
        if running_var is not None:
            running_var[:] = (1 - momentum) * running_var + np.multiply(momentum, var.data)

    else:
        assert (
                running_mean is not None and running_var is not None
        ), "Running{mean,var} are required during evaluation!"

        mean = running_mean
        var = running_var

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


def batch_norm_2d(
    input: Array,
    running_mean: Optional[np.ndarray] = None,
    running_var: Optional[np.ndarray] = None,
    weight: Optional[Array] = None,
    bias: Optional[Array] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Array:
    assert input.ndim == 4

    result = input
    if training:
        mean = result.mean(axis=(0, 2, 3))
        var = result.var(axis=(0, 2, 3))

        # update stats if given
        if running_mean is not None:
            running_mean[:] = (1 - momentum) * running_mean + np.multiply(momentum, mean.data)
        if running_var is not None:
            running_var[:] = (1 - momentum) * running_var + np.multiply(momentum, var.data)

    else:
        assert (
                running_mean is not None and running_var is not None
        ), "Running{mean,var} are required during evaluation!"

        mean = running_mean
        var = running_var

    mean = mean[..., np.newaxis, np.newaxis]
    var = var[..., np.newaxis, np.newaxis]

    # normalize
    result = result - mean
    if isinstance(var, Variable):
        denom = (var + eps).sqrt()
    else:
        denom = np.sqrt(var + eps)
    result = result / denom

    if weight is not None:
        result = result * weight[..., np.newaxis, np.newaxis]
    if bias is not None:
        result = result + bias[..., np.newaxis, np.newaxis]

    return result


# the implementation below is more efficient. However, for correctness reasons
# the simpler version above is used, which uses the base already implemented operators
# testing batch norm is difficult because of the difference in how numpy does
# its reduction operators on arrays.
# only for 2d
if MORE_EFFICIENT:
    def _batch_norm_2d(
        input: np.ndarray,  # no Variable!
        running_mean: Optional[np.ndarray] = None,
        running_var: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-05,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Actual implementation.
        """
        assert input.ndim == 4

        result = np.copy(input.data)
        if training:
            mean = result.mean(axis=(0, 2, 3))
            var = result.var(axis=(0, 2, 3))

            # update stats if given
            if running_mean is not None and running_var is not None:
                running_mean[:] = (1 - momentum) * running_mean + momentum * mean
                running_var[:] = (1 - momentum) * running_var + momentum * var

        else:
            assert (
                running_mean is not None and running_var is not None
            ), "Running{mean,var} are required during evaluation!"

            mean = running_mean
            var = running_var

        mean = mean[..., np.newaxis, np.newaxis]
        var = var[..., np.newaxis, np.newaxis]

        # normalize
        result = result - mean
        denom = np.sqrt(var + eps)
        result = result / denom

        if weight is not None:
            result = result * weight[..., np.newaxis, np.newaxis]
        if bias is not None:
            result = result + bias[..., np.newaxis, np.newaxis]

        return result, (mean, var, denom)


    class BatchNorm2dBackward(Operator):
        def __init__(
            self,
            input: Variable,
            mean: np.ndarray,
            var: np.ndarray,
            denom: np.ndarray,  # the sqrt(var + eps) value
            weight: Optional[Variable] = None,
            bias: Optional[Variable] = None,
        ):
            # variables
            self.input = input
            self.weight = weight
            self.bias = bias

            # not variables
            self.denom = denom
            self.mean = mean
            self.var = var

        def get_inputs(self) -> list[Variable]:
            inputs = [self.input]
            if self.weight is not None:
                inputs += [self.weight]
            if self.bias is not None:
                inputs += [self.bias]
            return inputs

        def backprop(self, out_grad: np.ndarray) -> None:
            #  The backward pass is based on https://kevinzakka.github.io/2016/09/14/batch_normalization/
            if self.input.requires_grad:
                dxhat = np.multiply(out_grad, self.weight.view(np.ndarray)[None, ..., None, None])

                N = np.prod([self.input.shape[i] for i in range(4) if i != 1])
                input_grad = (
                    (
                        N * dxhat
                        - dxhat.sum(axis=(0, 2, 3), keepdims=True)
                        - np.multiply(
                            self.input.data,
                            np.multiply(dxhat, self.input.data).sum(axis=(0, 2, 3), keepdims=True),
                        )
                    )
                    / N
                    / self.denom
                )
                self._update_grad(self.input, input_grad)

            if self.weight.requires_grad:
                weight_grad = np.multiply(out_grad, self.input.data).sum(axis=(0, 2, 3))
                self._update_grad(self.weight, weight_grad)

            if self.bias.requires_grad:
                bias_grad = out_grad.sum(axis=(0, 2, 3))
                self._update_grad(self.bias, bias_grad)


    def _ensure_is_variable_or_None(x: Union[Array, None]) -> Optional[Variable]:
        if x is not None:
            return Variable.ensure_is_variable(x)


    def batch_norm_2d(
        input: Array,
        running_mean: Optional[np.ndarray] = None,
        running_var: Optional[np.ndarray] = None,
        weight: Optional[Array] = None,
        bias: Optional[Array] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-05,
    ) -> Array:
        """
        Batch Norm function.

        :param input: The input. Should be a 4D array.
        :type input: Array
        :param running_mean: The running mean to update/use.
        :type running_mean: None | np.ndarray
        :param running_var: The running var to update/use.
        :type running_var: None | np.ndarray
        :param weight: The weight of the affine transform.
        :type weight: None | Array
        :param bias: The bias of the affine transform.
        :type bias: None | Array
        :param training: Whether it's training mode or eval mode.
        :type training: bool
        :param momentum: The momentum to use for running stats.
        :type momentum: float | int
        :param eps: The epsilon term to use.
        :type eps: float | int
        :return: batch normed input
        :rtype: Array
        """
        w = None if weight is None else weight.view(np.ndarray)
        b = None if bias is None else bias.view(np.ndarray)
        result_data, cache = _batch_norm_2d(input, running_mean, running_var, w, b, training, momentum, eps)

        if any(isinstance(obj, Variable) and obj.requires_grad for obj in [input, weight, bias]):
            input = Variable.ensure_is_variable(input)
            if weight is not None:
                weight = Variable.ensure_is_variable(weight)
            if bias is not None:
                bias = Variable.ensure_is_variable(bias)

            mean, var, denom = cache
            result = Variable(
                result_data,
                requires_grad=True,
                retains_grad=False,
                grad_fn=BatchNorm2dBackward(input, mean, var, denom, weight, bias)
            )

        else:
            result = result_data

        return result




