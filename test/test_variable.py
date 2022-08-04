import warnings

import pytest

import torch  # for comparing
import numpy as np
from simple_autograd import Variable
import simple_autograd.operations as O
import simple_autograd.backprop as B


def get_var():
    return Variable(np.zeros(1))


SHAPE = (3, 4, 7)


class TestBasic:
    """
    Tests for basic stuff.
    """
    def test_defaults(self):
        v = get_var()

        assert v.requires_grad is True
        assert v._grad is None
        assert v.grad is None

        assert v.retains_grad is True

    def test_default_grad_fn(self):
        v = get_var()

        assert isinstance(v.grad_fn, O.DoNothingBackward)

    def test_grad_no_warning(self):
        v = get_var()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = v.grad

    def test_grad_nonleaf_warning(self):
        v = get_var()

        with pytest.warns(UserWarning):
            v = v + 2  # do some random operation
            _ = v.grad

    def test_init_grad(self):
        zeros = np.zeros(SHAPE)
        v = Variable(zeros)

        v.init_grad()

        assert v.grad is not None
        assert v.grad.shape == SHAPE
        assert np.array_equal(v.grad, zeros)

    def test_retain_grad_after(self):
        v = get_var()
        v = v * 2

        assert v.retains_grad is False


EXPRESSIONS = [
  f"{unary_op}a"
  for unary_op in ["", "+", "-"]
] + [
  f"a{binary_op}b"
  for binary_op in ["+", "-", "*", "/", "%", "**"]
] + [
  # reversed a and b (one is a variable, one isn't)
  f"b{binary_op}a"
  for binary_op in ["+", "-", "*", "/", "**"]
] + [
  f"a.{f_op}()"
  for f_op in ["relu", "sum", "min", "max", "mean", "sqrt"]
] + [
  f"a.{binary_f_op}(b)"
  for binary_f_op in ["max_between", "min_between"]
] + [
  "a[idx]"
]


class TestOperatorsResultsAreVariables:
    @pytest.mark.parametrize("expr", EXPRESSIONS)
    def test(self, expr):
        a = Variable(np.zeros(10) + 2)
        b = 30  # some constant
        idx = slice(3, 4)

        result = eval(expr)
        assert isinstance(result, Variable)
        assert hasattr(result, "grad_fn")


class TestOperators:
    @pytest.mark.parametrize("expr", filter(lambda e: e not in [
        "+a",
        "a.min_between(b)",
        "a.max_between(b)",  # not defined for Tensors
    ], EXPRESSIONS))
    def test1(self, expr):
        a = Variable(np.zeros((3, 3)) + 2)
        a_torch = torch.tensor(np.zeros((3, 3)) + 2)
        a_torch.requires_grad = True

        b = 30  # some constant
        idx = slice(3, 5)

        result = eval(expr)
        result.sum().backward()

        result_torch = eval(expr.replace("a", "a_torch", 1))
        result_torch.sum().backward()

        np.testing.assert_array_equal(a.grad, a_torch.grad.numpy())


