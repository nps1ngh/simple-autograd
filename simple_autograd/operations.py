from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Tuple, Union

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
        var.init_grad()

        grad = np.atleast_1d(grad)  # make sure it's an array

        # sum reduce if needed
        shape_diff = grad.ndim - var.grad.ndim
        if shape_diff > 0:
            grad = grad.sum(axis=tuple(range(shape_diff)))
        if grad.shape != var.grad.shape:
            # if still not the same, then identify where broadcasting
            # happened and sum-reduce if needed
            # (only when grad.shape[i] > var.grad.shape[i])
            reduce_axis = tuple(
                i
                for i in range(min(grad.ndim, var.grad.ndim))
                if grad.shape[i] > var.grad.shape[i] and (var.grad.shape[i] == 1)
            )
            grad = grad.sum(axis=reduce_axis, keepdims=True)

        # update
        var.grad += grad

    @abc.abstractmethod
    def get_inputs(self) -> list[variable.Variable]:
        raise NotImplementedError


# for leaf nodes
class DoNothingBackward(Operator):
    # ensure that there is only one such instance
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

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


class NaryOperator(Operator, abc.ABC):
    def __init__(self, *variables: variable.Variable):
        self.variables = variables

    def get_inputs(self):
        return self.variables


class NegBackward(UnaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = np.negative(out_grad)
            self._update_grad(self.input, input_grad)


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


class DivBackward(NonCommutativeBinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            a_grad = out_grad / self.b.data
            self._update_grad(self.a, a_grad)

        if self.b.requires_grad:
            b_grad = -out_grad * self.a.data / np.square(self.b.data)  # -1/x^2
            self._update_grad(self.b, b_grad)


class ModBackward(NonCommutativeBinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            self._update_grad(self.a, out_grad)  # same as a - b

        if self.b.requires_grad:
            raise RuntimeError("the derivative for 'b' is not implemented!")


class MatMulBackward(NonCommutativeBinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            b = np.asarray(self.b.data)
            if b.ndim > 1:
                b = np.swapaxes(b, -1, -2)
            a_grad = out_grad @ b
            self._update_grad(self.a, a_grad)

        if self.b.requires_grad:
            a = np.asarray(self.a.data)
            if a.ndim > 1:
                a = np.swapaxes(a, -1, -2)
            b_grad = a @ out_grad
            self._update_grad(self.b, b_grad)


class ScalarProductBackward(NonCommutativeBinaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        assert out_grad.ndim <= 1, "outputs' incoming gradient should be 1-dim"

        if self.a.requires_grad:
            b = np.asarray(self.b.data)
            if b.ndim > 1:
                b = np.swapaxes(b, -1, -2)
            a_grad = out_grad * b
            self._update_grad(self.a, a_grad)

        if self.b.requires_grad:
            a = np.asarray(self.a.data)
            if a.ndim > 1:
                a = np.swapaxes(a, -1, -2)
            b_grad = out_grad * a
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
            a_grad = (
                out_grad
                * self.b.data
                * np.power(self.a.data, np.subtract(self.b.data, 1))
            )
            self._update_grad(self.a, a_grad)

        if self.b.requires_grad:
            b_grad = out_grad * np.multiply(self.output.data, np.log(self.a.data))
            self._update_grad(self.b, b_grad)


class MinimumMaximumBackward(BinaryOperator):
    def __init__(
        self, a: variable.Variable, b: variable.Variable, choose_left: np.ndarray
    ):
        super().__init__(a, b)
        self.choose_left = choose_left

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.a.requires_grad:
            a_grad = np.where(self.choose_left, out_grad, 0)
            self._update_grad(self.a, a_grad)

        if self.b.requires_grad:
            b_grad = np.where(self.choose_left, 0, out_grad)
            self._update_grad(self.b, b_grad)


# -------------------------------------------------------------
# Elementwise Operators
# -------------------------------------------------------------
class SqrtBackward(UnaryOperator):
    def __init__(self, input: variable.Variable, output: np.ndarray):
        super().__init__(input)
        self.output = output

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = out_grad / 2 / self.output
            self._update_grad(self.input, input_grad)


class ReLUBackward(UnaryOperator):
    def __init__(self, input: variable.Variable, chosen: np.ndarray):
        super().__init__(input)
        self.chosen = chosen

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = np.where(self.chosen, out_grad, 0)
            self._update_grad(self.input, input_grad)


class ExpBackward(UnaryOperator):
    def __init__(self, input: variable.Variable, out: variable.Variable):
        super().__init__(input)
        self.out = out

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = np.multiply(out_grad, self.out.data)
            self._update_grad(self.input, input_grad)


class SinBackward(UnaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = out_grad * np.cos(self.input.data)
            self._update_grad(self.input, input_grad)


class CosBackward(UnaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = -out_grad * np.sin(self.input.data)
            self._update_grad(self.input, input_grad)


class SigmoidBackward(UnaryOperator):
    def __init__(
        self,
        input: variable.Variable,
        exp_m_x: variable.Variable,
        out: variable.Variable,
    ):
        super().__init__(input)
        self.exp_m_x = exp_m_x
        self.out = out

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = np.multiply(out_grad, self.exp_m_x.data) * np.square(
                self.out.data
            )
            self._update_grad(self.input, input_grad)


class LogBackward(UnaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = np.divide(out_grad, self.input.data)
            self._update_grad(self.input, input_grad)


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

        # a bit hacky but works
        if axis is not None:
            if isinstance(axis, tuple):
                axis = tuple(a + input.ndim if a < 0 else a for a in axis)
            else:
                axis = axis + input.ndim if axis < 0 else axis
        self.axis = axis
        self.keepdims = keepdims

    def __repr__(self):
        if self.axis is not None:
            axis = np.atleast_1d(self.axis)
            return f"{super().__repr__()[:-1]}{''.join(map(str, axis))}>"
        else:
            return super().__repr__()


class SumBackward(ReductionOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            if self.axis is not None and not self.keepdims:
                shape = self.input.shape
                axis = set(np.atleast_1d(self.axis))
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
                axis = set(np.atleast_1d(self.axis))
                # replace reduction dims with 1
                shape = tuple(1 if i in axis else dim for i, dim in enumerate(shape))
                out_grad = out_grad.reshape(shape)
            input_grad = out_grad / np.prod(
                list(self.input.shape[i] for i in np.atleast_1d(self.axis))
            )
        else:
            # no axis
            input_grad = out_grad / np.prod(self.input.shape)

        self._update_grad(self.input, input_grad)


class VarBackward(ReductionOperator):
    def __init__(self, input, axis, unbiased, keepdims):
        super().__init__(input, axis, keepdims)
        self.unbiased = unbiased

    def backprop(self, out_grad: np.ndarray) -> None:
        if not self.input.requires_grad:
            return

        if self.axis is not None:
            if not self.keepdims:  # only reshape if needed
                shape = self.input.shape
                axis = set(np.atleast_1d(self.axis))
                # replace reduction dims with 1
                shape = tuple(1 if i in axis else dim for i, dim in enumerate(shape))
                out_grad = out_grad.reshape(shape)
            n = np.prod(list(self.input.shape[i] for i in np.atleast_1d(self.axis)))

        else:
            # no axis
            n = np.prod(self.input.shape)

        denom = n - 1 if self.unbiased else n
        input_grad = (
            2
            * out_grad
            * np.subtract(
                self.input.data, np.mean(self.input.data, self.axis, keepdims=True)
            )
            / denom
        )

        self._update_grad(self.input, input_grad)


class MinMaxRBackward(ReductionOperator):
    def __init__(
        self,
        input: variable.Variable,
        idx: np.ndarray,
        axis: Optional[Union[int, Tuple[int]]],
        keepdims: bool,
    ):
        super().__init__(input, axis, keepdims)
        self.idx = idx

    def backprop(self, out_grad: np.ndarray) -> None:
        if self.input.requires_grad:
            input_grad = np.zeros_like(self.input.data)
            if self.axis is None:
                input_grad[self.idx] = out_grad.item()  # output should be a "scalar"
                input_grad /= len(self.idx[0])
            else:
                np.put_along_axis(
                    input_grad,
                    np.expand_dims(self.idx, self.axis),
                    out_grad if self.keepdims else np.expand_dims(out_grad, self.axis),
                    self.axis,
                )

            self._update_grad(self.input, input_grad)


class TransposeBackward(UnaryOperator):
    def __init__(self, input: variable.Variable, source: int, destination: int):
        super().__init__(input)
        self.source = source
        self.destination = destination

    def backprop(self, out_grad: np.ndarray) -> None:
        input_grad = np.swapaxes(out_grad, self.destination, self.source)
        self.input.init_grad()
        assert (
            input_grad.shape == self.input.grad.shape
        ), f"Unequal shapes! {input_grad.shape=} and {self.input.grad.shape=}"
        self.input.grad += input_grad


class ReshapeBackward(UnaryOperator):
    def backprop(self, out_grad: np.ndarray) -> None:
        input_grad = np.reshape(out_grad, self.input.shape)
        self.input.init_grad()
        self.input.grad += input_grad


# -------------------------------------------------------------
# Others
# -------------------------------------------------------------
class IndexingBackward(UnaryOperator):
    def __init__(self, input: variable.Variable, item_idx):
        super().__init__(input)
        self.item_idx = item_idx

    def backprop(self, out_grad: np.ndarray) -> None:
        self.input.init_grad()

        # yes this is all you need to update the grad
        np.add.at(self.input.grad, self.item_idx, out_grad)


class ConcatenateBackward(NaryOperator):
    def __init__(self, *variables, axis=0):
        super().__init__(*variables)
        self.axis = axis

    def backprop(self, out_grad: np.ndarray) -> None:
        axis = self.axis
        i = 0  # position on axis
        og = out_grad.swapaxes(0, axis)
        for var in self.variables:
            j = i + var.shape[axis]
            if var.requires_grad:
                self._update_grad(var, og[i:j].swapaxes(axis, 0))

            i = j
