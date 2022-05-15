from typing import Optional

import numpy as np

from . import backprop
from . import operations


class Variable(np.ndarray):
    def __new__(cls, data: np.ndarray, *args, **kwargs):
        assert data.dtype == "float", "only floats supported"
        data = np.asarray(data).view(cls)
        return data

    def __init__(
        self,
        _: np.ndarray,
        requires_grad: bool = True,
        grad_fn: operations.Operator = operations.DoNothingBackward(),
    ):
        self.requires_grad: bool = requires_grad

        self.grad: Optional[np.ndarray] = None
        self.grad_fn = grad_fn

    def __repr__(self):
        result = super().__repr__()[:-1]  # last is ')'

        if self.grad_fn is not None and not isinstance(
            self.grad_fn, operations.DoNothingBackward
        ):
            result += f", grad_fn={self.grad_fn}"

        result += ")"
        return result

    @staticmethod
    def _ensure_is_variable(other, matrix: bool = False) -> "Variable":
        if not isinstance(other, Variable):
            if not isinstance(other, np.ndarray):
                # convert to array
                other = float(other)
                other = np.atleast_2d(other) if matrix else np.atleast_1d(other)

            other = Variable(other, requires_grad=False)

        return other

    def backward(self) -> None:
        """
        Back-propagate.
        Works only for scalars!
        """
        backprop.backward(self)

    def detach(self) -> "Variable":
        """
        Returns the current variable but with gradient removed.
        """
        return Variable(self, requires_grad=self.requires_grad)

    # -------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------
    def __add__(self, other) -> "Variable":
        other = self._ensure_is_variable(other)

        result_data = super().__add__(other)  # other.data + self.data
        result = Variable(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=operations.AddBackward(self, other),
        )

        return result

    def __radd__(self, other) -> "Variable":
        return self.__add__(other)  # is commutative

    def __sub__(self, other, reverse=False) -> "Variable":
        other = self._ensure_is_variable(other)

        if not reverse:
            result_data = super().__sub__(other)
        else:
            # other WILL be a variable so fine
            result_data = super(type(other), other).__sub__(self)

        result = Variable(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=operations.SubBackward(self, other, reverse=reverse),
        )

        return result

    def __rsub__(self, other) -> "Variable":
        return self.__sub__(other, reverse=True)

    def __mul__(self, other) -> "Variable":
        other = self._ensure_is_variable(other)

        result_data = super().__mul__(other)  # other.data * self.data
        result = Variable(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=operations.MulBackward(self, other),
        )

        return result

    def __rmul__(self, other) -> "Variable":
        return self.__mul__(other)  # is commutative

    def __matmul__(self, other, reverse=False) -> "Variable":
        other = self._ensure_is_variable(other, matrix=True)

        if not reverse:
            result_data = super().__matmul__(other)
        else:
            result_data = super(type(other), other).__matmul__(self)
        result = Variable(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            grad_fn=operations.MatMulBackward(self, other, reverse=reverse),
        )

        return result

    def __rmatmul__(self, other) -> "Variable":
        return self.__matmul__(other, reverse=True)

    def __pow__(self, other, reverse=True):
        other = self._ensure_is_variable(other)

        if not reverse:
            result_data = super().__pow__(other)
        else:
            result_data = super(type(other), other).__pow__(self)

        result = Variable(
            result_data,
            requires_grad=self.requires_grad,
            grad_fn=operations.PowBackward(self, other, reverse=reverse),
        )

        return result

    def __rpow__(self, other):
        return self.__pow__(other, reverse=True)

    # -------------------------------------------------------------
    # Reduction Operators
    # -------------------------------------------------------------
    def sum(self, axis=None, keepdims=False, initial=None, *args, **kwargs):
        # TODO: incorporate other args
        result_data = super().sum(axis, keepdims, initial)

        return Variable(
            result_data,
            requires_grad=self.requires_grad,
            grad_fn=operations.SumBackward(self, axis=axis, keepdims=keepdims),
        )

    def max(self, axis=None, keepdims=False, initial=None, *args, **kwargs):
        result_data_idx = super().argmax()

        return Variable(
            np.asarray(self.data)[result_data_idx],
            requires_grad=self.requires_grad,
            grad_fn=operations.MaxRBackward(self, idx=result_data_idx),
        )

    # -------------------------------------------------------------
    # Elementwise functional Operators
    # -------------------------------------------------------------
    def sqrt(self):
        return self.__pow__(0.5)

    def relu(self):
        raise NotImplementedError

    # -------------------------------------------------------------
    # Not implemented
    # -------------------------------------------------------------
    def __mod__(self, other):
        return NotImplemented

    def __rmod__(self, other):
        return NotImplemented

    # -------------------------------------------------------------
    # Others
    # -------------------------------------------------------------
    def __hash__(self):
        # needed for top-sorting
        # this is fine since we only care about individual nodes in the
        # graph, which this facilitates
        return id(self)
