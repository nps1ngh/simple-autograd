import warnings
from typing import Optional

import numpy as np

from . import backprop
from . import operations


class Variable(np.ndarray):
    # see https://numpy.org/doc/stable/user/basics.subclassing.html
    def __new__(cls, input_array,
                requires_grad=True, retains_grad=True, grad_fn=operations.DoNothingBackward(),
                ):
        obj = np.asarray(input_array).view(cls)

        obj.requires_grad = requires_grad
        obj.retains_grad = retains_grad and requires_grad

        obj._grad = None
        obj.grad_fn = grad_fn

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        assert obj.dtype in [float, np.float32, np.float16], f"only floats supported! but was {obj.dtype}"

        self.requires_grad: bool = getattr(obj, "requires_grad", True)
        self.retains_grad: bool = getattr(obj, "retains_grad", True) and self.requires_grad

        self._grad: Optional[np.ndarray] = None
        self.grad_fn = getattr(obj, "grad_fn", operations.DoNothingBackward())

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
            other = Variable(other, requires_grad=False, retains_grad=False)

        return other

    def backward(self) -> None:
        """
        Back-propagate.
        Works only for scalars!
        """
        backprop.backward(self)

    def detach(self) -> "Variable":
        """
        Returns the current variable but completely detached from all
        gradient computation.
        """
        return Variable(self, requires_grad=False, retains_grad=False)

    def retain_grad(self, value=True) -> None:
        """
        Retain gradients during backpropagation.
        """
        self.retains_grad = value

    def zero_grad(self) -> None:
        """
        Set gradients if available to 0.
        """
        if self.grad is not None:
            self.grad.fill(0)


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
            retains_grad=False,
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
            # we cannot use super() because the dtype is int64
            # but our Variable class doesn't support it
            # result_data_idx = super().argmax(axis=axis)
            result_data_idx = np.argmax(self.data, axis=axis)
        else:
            result_data_idx = np.argmin(self.data, axis=axis)

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
                _axis = axis if axis >= 0 else axis + self.ndim
                result_data = result_data.reshape(self.shape[:_axis] + self.shape[_axis + 1:])

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

    def var(self, axis=None, unbiased=True, keepdims=False, *args, **kwargs):
        keepdims = keepdims or kwargs.get("keepdim", False)
        ddof = 1 if unbiased else 0
        result_data = super().var(axis=axis, ddof=ddof, keepdims=keepdims)

        result = self._create_variable(
            data=result_data,
            grad_fn=operations.VarBackward(self, axis=axis, unbiased=unbiased, keepdims=keepdims),
        )
        return result

    def std(self, axis=None, unbiased=True, keepdims=False, *args, **kwargs):
        return self.var(axis, unbiased, keepdims, *args, **kwargs).sqrt()

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

    def exp(self):
        result_data = np.exp(self.data)
        result = self._create_variable(
            data=result_data,
            grad_fn=operations.ExpBackward(self, out=result_data),
        )

        return result

    def sin(self):
        result_data = np.sin(self.data)
        result = self._create_variable(
            data=result_data,
            grad_fn=operations.SinBackward(self),
        )

        return result

    def cos(self):
        result_data = np.cos(self.data)
        result = self._create_variable(
            data=result_data,
            grad_fn=operations.CosBackward(self),
        )

        return result

    def sigmoid(self):
        exp_m_x = np.exp(np.negative(self.data))
        result_data = np.reciprocal(1 + exp_m_x)
        result = self._create_variable(
            data=result_data,
            grad_fn=operations.SigmoidBackward(self, exp_m_x, result_data),
        )

        return result

    def softmax(self, axis):
        # TODO: dedicated faster backward pass
        x = self - self.max(axis, keepdims=True)

        exp = x.exp()
        denom = exp.sum(axis, keepdims=True)
        out = exp / denom

        return out

    def log(self):
        result_data = np.log(self.data)
        result = self._create_variable(
            data=result_data,
            grad_fn=operations.LogBackward(self),
        )

        return result

    def log_softmax(self, axis):
        # based on https://stackoverflow.com/a/61570752/10614892
        c = self.max(axis, keepdims=True)
        x = self - c
        logsumexp = x.exp().sum(axis, keepdims=True).log()

        return x - logsumexp

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

    def concat(self, *others, axis=0):
        others = [self.ensure_is_variable(other) for other in others]
        result_data = np.concatenate([self.data] + [other.data for other in others], axis=axis)
        result = self._create_variable(
            data=result_data, grad_fn=operations.ConcatenateBackward(self, *others, axis=axis),
        )
        return result

    # -------------------------------------------------------------
    # Shape methods
    # -------------------------------------------------------------
    def swapaxes(self, axis1: int, axis2: int):
        """
        The same as `.transpose`.
        """
        return self.transpose(axis1, axis2)

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

    def reshape(self, shape, order='C'):
        result_data = np.reshape(self.data, shape)
        result = self._create_variable(
            data=result_data,
            grad_fn=operations.ReshapeBackward(self),
        )

        return result

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
