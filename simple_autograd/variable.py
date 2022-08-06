from typing import Optional
import warnings

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
        retain_grad: bool = True,
        grad_fn: operations.Operator = operations.DoNothingBackward(),
    ):
        self.requires_grad: bool = requires_grad
        self.retains_grad: bool = retain_grad and requires_grad

        self._grad: Optional[np.ndarray] = None
        self.grad_fn = grad_fn

    @property
    def grad(self):
        if self._grad is None and not isinstance(
                self.grad_fn, operations.DoNothingBackward
        ):
            warnings.warn(
                "If you want to access the .grad attribute of a non-leaf variable "
                "then you should call .retain_grad() on it before calling "
                ".backward() !"
            )
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def init_grad(self) -> None:
        """
        Initializes gradient if needed.
        """
        if self._grad is None:
            self._grad = np.zeros_like(self.data)  # init

    def __repr__(self):
        result = repr(super().view(np.ndarray))  # super().__repr__ leads to problems
        result = result[:-1]  # last is ')'

        if self.grad_fn is not None:
            if not isinstance(self.grad_fn, operations.DoNothingBackward):
                result += f", grad_fn={self.grad_fn}"
            else:
                result += f", requires_grad={self.requires_grad}"

        result += ")"
        return result

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        # needed for top-sorting
        # this is fine since we only care about individual nodes in the
        # graph, which this facilitates
        return id(self)

    @staticmethod
    def ensure_is_variable(other, matrix: bool = False) -> "Variable":
        if not isinstance(other, Variable):
            if not isinstance(other, np.ndarray):
                # convert to array
                other = float(other)
                other = np.atleast_2d(other) if matrix else np.atleast_1d(other)

            other = np.asarray(other, float)  # okay since self is float
            other = Variable(other, requires_grad=False, retain_grad=False)

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

    def retain_grad(self, value=True) -> None:
        """
        Retain gradients during backpropagation.
        """
        self.retains_grad = value

    # -------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------
    def _create_variable(self, data, grad_fn, other=None) -> "Variable":
        """
        Helper function for creating variable inside.
        :param data: The data of the new variable.
        :type data: np.ndarray or memoryview
        :param grad_fn: The gradient function.
        :type grad_fn: operations.Operation
        :param other: The other variable. Only needed for binary ops.
        :type other: Variable
        :return: the resultant variable
        :rtype: Variable
        """
        other_requires_grad = False if other is None else other.requires_grad

        result = Variable(
            data,
            requires_grad=self.requires_grad or other_requires_grad,
            retain_grad=False,
            grad_fn=grad_fn,
        )
        return result

    def __pos__(self) -> "Variable":
        return self  # no need to do anything

    def __neg__(self) -> "Variable":
        result_data = super().__neg__()
        result = self._create_variable(
            data=result_data, grad_fn=operations.NegBackward(self),
        )
        return result

    def __add__(self, other) -> "Variable":
        other = self.ensure_is_variable(other)

        result_data = np.add(self.data, other.data)
        result = self._create_variable(
            data=result_data, other=other, grad_fn=operations.AddBackward(self, other),
        )

        return result

    def __radd__(self, other) -> "Variable":
        return self.__add__(other)  # is commutative

    def __sub__(self, other, reverse=False) -> "Variable":
        other = self.ensure_is_variable(other)

        if not reverse:
            result_data = super().__sub__(other)
        else:
            # other WILL be a variable so fine
            result_data = super(type(other), other).__sub__(self)

        result = self._create_variable(
            data=result_data,
            other=other,
            grad_fn=operations.SubBackward(self, other, reverse=reverse),
        )

        return result

    def __rsub__(self, other) -> "Variable":
        return self.__sub__(other, reverse=True)

    def __mul__(self, other) -> "Variable":
        other = self.ensure_is_variable(other)

        result_data = super().__mul__(other)  # other.data * self.data
        result = self._create_variable(
            data=result_data, other=other, grad_fn=operations.MulBackward(self, other),
        )

        return result

    def __rmul__(self, other) -> "Variable":
        return self.__mul__(other)  # is commutative

    def __truediv__(self, other, reverse=False) -> "Variable":
        other = self.ensure_is_variable(other)

        if not reverse:
            result_data = super().__truediv__(other)
        else:
            result_data = super(type(other), other).__truediv__(self)
        result = self._create_variable(
            data=result_data, other=other, grad_fn=operations.DivBackward(self, other, reverse=reverse),
        )

        return result

    def __rtruediv__(self, other) -> "Variable":
        return self.__truediv__(other, reverse=True)

    def __mod__(self, other) -> "Variable":
        other = self.ensure_is_variable(other)

        result_data = super().__mod__(other)
        result = self._create_variable(
            data=result_data, other=other, grad_fn=operations.ModBackward(self, other, reverse=False),
        )

        return result

    def __matmul__(self, other, reverse=False) -> "Variable":
        other = self.ensure_is_variable(other, matrix=True)

        if not reverse:
            result_data = super().__matmul__(other)
        else:
            result_data = super(type(other), other).__matmul__(self)

        if self.ndim == 1 and other.ndim == 1:
            backward_op = operations.ScalarProductBackward
        else:
            backward_op = operations.MatMulBackward

        result = self._create_variable(
            data=result_data,
            other=other,
            grad_fn=backward_op(self, other, reverse=reverse),
        )

        return result

    def __rmatmul__(self, other) -> "Variable":
        return self.__matmul__(other, reverse=True)

    def __pow__(self, other, reverse=False):
        other = self.ensure_is_variable(other)

        if not reverse:
            result_data = super().__pow__(other)
        else:
            result_data = super(type(other), other).__pow__(self)

        result = self._create_variable(
            data=result_data,
            other=other,
            grad_fn=operations.PowBackward(
                self, other, reverse=reverse, output=result_data
            ),
        )

        return result

    def __rpow__(self, other):
        return self.__pow__(other, reverse=True)

    def _minimummaximum(self, other, do_max=True):
        other = self.ensure_is_variable(other)

        if do_max:
            choose_left = np.greater(self.data, other.data)
        else:
            choose_left = np.less(self.data, other.data)

        result_data = np.where(choose_left, self.data, other.data)
        result = self._create_variable(
            data=result_data,
            other=other,
            grad_fn=operations.MinimumMaximumBackward(self, other, choose_left),
        )

        return result

    def maximum(self, other):
        return self._minimummaximum(other)

    def minimum(self, other):
        return self._minimummaximum(other, do_max=False)

    # -------------------------------------------------------------
    # Reduction Operators
    # -------------------------------------------------------------
    def sum(self, axis=None, keepdims=False, initial=0, *args, **kwargs):
        keepdims = keepdims or kwargs.get("keepdim", False)
        result_data = super().sum(axis=axis, keepdims=keepdims, initial=initial)

        result = self._create_variable(
            data=result_data,
            grad_fn=operations.SumBackward(self, axis=axis, keepdims=keepdims),
        )
        return result

    def _minmax(self, axis=None, keepdims=False, do_max=True):
        if do_max:
            result_data_idx = super().argmax(axis=axis)
        else:
            result_data_idx = super().argmin(axis=axis)

        if axis is None:
            result_data = np.asarray(self.data).ravel()[result_data_idx]
            result_data_idx = np.where(np.equal(self.data, result_data))
            if keepdims:
                result_data = np.expand_dims(result_data, tuple(range(self.ndim)))
            else:
                result_data = np.atleast_1d(result_data)  # scalar
        else:
            result_data = np.take_along_axis(
                np.asarray(self.data), np.expand_dims(result_data_idx, axis), axis
            )
            if not keepdims:
                result_data = result_data.reshape(self.shape[:axis] + self.shape[axis + 1:])

        result = self._create_variable(
            data=result_data,
            grad_fn=operations.MinMaxRBackward(
                self, axis=axis, keepdims=keepdims, idx=result_data_idx
            ),
        )
        return result

    def max(self, axis=None, keepdims=False, *args, **kwargs):
        keepdims = keepdims or kwargs.get("keepdim", False)
        return self._minmax(axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False, *args, **kwargs):
        keepdims = keepdims or kwargs.get("keepdim", False)
        return self._minmax(axis=axis, keepdims=keepdims, do_max=False)

    def mean(self, axis=None, keepdims=False, *args, **kwargs):
        keepdims = keepdims or kwargs.get("keepdim", False)
        result_data = super().mean(axis=axis, keepdims=keepdims)

        result = self._create_variable(
            data=result_data,
            grad_fn=operations.MeanBackward(self, axis=axis, keepdims=keepdims),
        )
        return result

    # -------------------------------------------------------------
    # Elementwise functional Operators
    # -------------------------------------------------------------
    def sqrt(self):
        result_data = np.sqrt(self.data)
        result = self._create_variable(
            data=result_data, grad_fn=operations.SqrtBackward(self, output=result_data),
        )
        return result

    def relu(self):
        choose = np.greater(self.data, 0)

        result_data = np.where(choose, self.data, 0)
        result = self._create_variable(
            data=result_data, grad_fn=operations.ReLUBackward(self, chosen=choose),
        )

        return result

    # -------------------------------------------------------------
    # Others
    # -------------------------------------------------------------
    def __getitem__(self, item) -> "Variable":
        item = np.index_exp[item]
        result_data = super().view(type=np.ndarray)[item]
        result = self._create_variable(
            data=result_data, grad_fn=operations.IndexingBackward(self, item),
        )
        return result

    def __setitem__(self, key, value):
        if self.requires_grad:
            raise RuntimeError(
                "Tried changing the value of an array that requires grad!"
            )
        else:
            super().__setitem__(key, value)

    # -------------------------------------------------------------
    # Shape methods
    # -------------------------------------------------------------
    def transpose(self, source: int, destination: int):
        """
        For transposing axes.

        Short note on design choice:
        We do not follow np.ndarray.transpose but instead
        torch.Tensor.transpose because then we have something to
        compare our implementation to.

        :param source: The source axis index.
        :type source: int
        :param destination: The target axis index.
        :type destination: int
        :return: The transposed array.
        :rtype: Variable
        """
        assert isinstance(source, int)
        assert isinstance(destination, int)
        assert -self.ndim <= source < self.ndim
        assert -self.ndim <= destination < self.ndim

        result_data = np.swapaxes(self.data, source, destination)
        result = self._create_variable(
            data=result_data,
            grad_fn=operations.TransposeBackward(self, source, destination),
        )

        return result

    def t(self):
        """
        Works just like torch.Tensor.t()
        Expects self to be <= 2D.
        Does a transpose if 2D. Otherwise, does nothing (for 0D and 1D).
        """
        assert self.ndim <= 2, f"Array should be <= 2D. Was {self.ndim}D"
        if self.ndim <= 1:
            return self
        else:
            return self.transpose(0, 1)

    # -------------------------------------------------------------
    # Some common methods which are not implemented.
    # Serves to warn the end-user.
    # np.ndarray has too many functions, so I'm not going to list
    # every not implemented one.
    # -------------------------------------------------------------
    def __rmod__(self, other):
        return NotImplemented

    def __floordiv__(self, other):
        return NotImplemented

    def __rfloordiv__(self, other):
        return NotImplemented

    def resize(self, new_shape, refcheck=True):
        raise NotImplementedError("Use .reshape instead!")

    @property
    def T(self):
        raise NotImplementedError("See .transpose()")


