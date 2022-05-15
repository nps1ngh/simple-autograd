from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Union, Tuple

import numpy as np

if TYPE_CHECKING:
    from . import variable


class Operator(abc.ABC):
    @abc.abstractmethod
    def backprop(self, out_grad: np.ndarray) -> None:
        """
        Back-propagates the gradient to parent variables in the computational graph.
        :param out_grad: The incoming output gradient.
        :type out_grad: np.ndarray
        """
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__}>"  # somewhat like pytorch

    @staticmethod
    def _update_grad(var: variable.Variable, grad: np.ndarray) -> None:
        """
        Updates the gradient of the given variable.
        Makes sure to initialize it if needed.
        Also, appropriately sum-reduces the `grad` if it's shape is larger.
        :param var: The variable whose gradient to update.
        :type var: variable.Variable
        :param grad: The update
        :type grad: np.ndarray
        """
        if var.grad is None:
            var.grad = np.zeros_like(var.data)  # init

        grad = np.atleast_1d(grad)  # make sure it's an array

        # sum reduce if needed
        shape_diff = len(grad.shape) - len(var.grad.shape)
        if shape_diff > 0:
            grad = grad.sum(axis=tuple(range(shape_diff)))

        # update
        var.grad += grad

    @abc.abstractmethod
    def get_inputs(self) -> list[variable.Variable]:
        raise NotImplementedError


# for leaf nodes
class DoNothingBackward(Operator):
    def backprop(self, out_grad: np.ndarray) -> None:
        pass  # leaf node, can't propagate further

    def get_inputs(self) -> list[variable.Variable]:
        return []


class UnaryOperator(Operator, abc.ABC):
    def __init__(self, input: variable.Variable):
        self.input = input

    def get_inputs(self) -> list[variable.Variable]:
        return [self.input]


class BinaryOperator(Operator, abc.ABC):
    def __init__(self, a: variable.Variable, b: variable.Variable):
        self.a = a
        self.b = b

    def get_inputs(self) -> list[variable.Variable]:
        return [self.a, self.b]


class NonCommutativeBinaryOperator(BinaryOperator, abc.ABC):
    def __init__(self, a: variable.Variable, b: variable.Variable, reverse: bool):
        if reverse:
            super().__init__(b, a)
        else:
            super().__init__(a, b)


class AddBackward(BinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            self._update_grad(self.a, out_grad)
        if self.b.requires_grad:
            self._update_grad(self.b, out_grad)


class SubBackward(NonCommutativeBinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            self._update_grad(self.a, out_grad)
        if self.b.requires_grad:
            # note the minus
            self._update_grad(self.b, -out_grad)


class MulBackward(BinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            a_grad = out_grad * self.b.data  # this is okay since out_grad is ndarray
            self._update_grad(self.a, a_grad)
        if self.b.requires_grad:
            b_grad = self.a.data * out_grad
            self._update_grad(self.b, b_grad)


class MatMulBackward(NonCommutativeBinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            a_grad = out_grad @ self.b.data.transpose((-1, -2))
            self._update_grad(self.a, a_grad)

        if self.b.requires_grad:
            b_grad = self.a.data.transpose((-1, -2)) @ out_grad
            self._update_grad(self.b, b_grad)


class PowBackward(NonCommutativeBinaryOperator):
    def __init__(
        self,
        a: variable.Variable,
        b: variable.Variable,
        reverse: bool,
        output: variable.Variable,
    ):
        super().__init__(a, b, reverse)
        self.output = output

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            # np.func can process memoryviews no problem
            a_grad = out_grad * (self.output * np.log(self.a.data))
            self._update_grad(self.a, a_grad)

        if self.b.requires_grad:
            b_grad = self.b.data * np.power(self.a.data, np.sub(self.b.data, 1))
            self._update_grad(self.b, b_grad)


# -------------------------------------------------------------
# Reduction Operators
# -------------------------------------------------------------


class ReductionOperator(UnaryOperator):
    def __init__(
        self,
        input: variable.Variable,
        axis: Optional[Union[int, Tuple[int]]],
        keepdims: bool,
    ):
        super().__init__(input)
        self.axis = axis
        self.keepdims = keepdims


class SumBackward(ReductionOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            if self.axis is not None and not self.keepdims:
                shape = self.input.shape
                axis = set(self.axis)
                # replace reduction dims with 1
                shape = tuple(1 if i in axis else dim for i, dim in enumerate(shape))
                input_grad = out_grad.reshape(shape)
            else:
                input_grad = out_grad  # is broadcastable

            self._update_grad(self.input, input_grad)


class MeanBackward(ReductionOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if not self.input.requires_grad:
            return

        if self.axis is not None:
            if not self.keepdims:  # only reshape if needed
                shape = self.input.shape
                axis = set(self.axis)
                # replace reduction dims with 1
                shape = tuple(
                    1 if i in axis else dim for i, dim in enumerate(shape)
                )
                out_grad = out_grad.reshape(shape)
            input_grad = out_grad / np.prod(self.axis)
        else:
            # no axis
            input_grad = out_grad / np.prod(self.input.shape)

        self._update_grad(self.input, input_grad)


class MaxRBackward(UnaryOperator):
    def __init__(self, input: variable.Variable, idx: np.ndarray):
        super().__init__(input)
        self.idx = idx

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            grad = out_grad * self.idx
            self._update_grad(self.input, grad)
