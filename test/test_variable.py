"""
Integration tests.
Check correctness with torch.
"""

import warnings
import random

import numpy as np
import pytest
import torch  # for comparing

import simple_autograd.operations as O
from simple_autograd import Variable


def get_var():
    return Variable(np.zeros(1))


SHAPES = [(4,), (100,), (2, 3), (3, 4, 5), (2, 3, 4, 5), (2, 3, 4, 1, 6), (2, 2, 2, 2, 2, 2, 2)]


@pytest.fixture(scope="module", autouse=True,
                params=SHAPES)
def shape(request):
    return request.param


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

    def test_init_grad(self, shape):
        zeros = np.zeros(shape)
        v = Variable(zeros)

        v.init_grad()

        assert v.grad is not None
        assert v.grad.shape == shape
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
                  for f_op in ["relu", "sqrt"]
              ] + [
                  f"a.{r_op}({args})"
                  for r_op in ["sum", "min", "max", "mean"]
                  for args in ["", "0", "0, keepdim=True"]
              ] + [
                  f"a.{binary_f_op}(b)"
                  for binary_f_op in ["maximum", "minimum"]
              ] + [
                  "a[idx]"
              ]


class TestOperatorsResultsAreVariables:
    @pytest.mark.parametrize("expr", EXPRESSIONS)
    def test(self, expr, shape):
        np.random.seed(42)
        a = Variable(np.random.randn(*shape))
        b = 30  # some constant
        idx = slice(3, 4)

        result = eval(expr)
        assert isinstance(result, Variable)
        assert hasattr(result, "grad_fn")
        assert hasattr(result, "requires_grad")


class TestOperators:
    @pytest.mark.parametrize("expr", filter(lambda e: e not in [
        "+a",
        "a.minimum(b)",
        "a.maximum(b)",  # not defined for Tensors
        "a[idx]",  # separate
    ], EXPRESSIONS))
    def test_straight_b_scalar(self, expr, shape):
        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b = 30  # some scalar

        result_torch = eval(expr.replace("a", "a_torch", 1))
        if isinstance(result_torch, tuple):  # for min max stuff
            result_torch = result_torch.values

        result = eval(expr)
        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())

        result_torch.sum().backward()
        result.sum().backward()

        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())

    @pytest.mark.parametrize("expr", filter(lambda e: "b" in e and e not in [
        "+a",
        "a.minimum(b)",
        "a.maximum(b)",
        "a[idx]",  # separate
        "a%b",  # not defined
    ], EXPRESSIONS))
    def test_reverse_b_scalar(self, expr, shape):
        np.random.seed(42)
        b_data = np.random.randn(*shape)
        b = Variable(b_data.copy())
        b_torch = torch.tensor(b_data.copy())
        b_torch.requires_grad = True

        a = 30  # some scalar

        result = eval(expr)
        result.sum().backward()

        result_torch = eval(expr.replace("b", "b_torch", 1))
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())

    @pytest.mark.parametrize("expr", filter(lambda e: "b" in e and e not in [
        "+a",
        "a.minimum(b)",
        "a.maximum(b)",  # not defined for Tensors
        "a[idx]",  # separate
    ], EXPRESSIONS))
    def test_b_array_same_shape_no_grad(self, expr, shape):
        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b_data = np.random.randn(*shape)
        b = b_data.copy()
        b_torch = torch.tensor(b_data.copy())

        result = eval(expr)
        result.sum().backward()

        expr_torch = expr.replace("a", "a_torch", 1).replace("b", "b_torch", 1)
        result_torch = eval(expr_torch)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())

    @pytest.mark.parametrize("expr", filter(lambda e: "b" in e and e not in [
        "+a",
        "a.minimum(b)",
        "a.maximum(b)",  # not defined for Tensors
        "a[idx]",  # separate
        "a%b",  # grad for other not defined
    ], EXPRESSIONS))
    def test_b_array_same_shape_with_grad(self, expr, shape):
        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b_data = np.random.randn(*shape)
        b = Variable(b_data.copy())
        b_torch = torch.tensor(b_data.copy())
        b_torch.requires_grad = True

        result = eval(expr)
        result.sum().backward()

        expr_torch = expr.replace("a", "a_torch", 1).replace("b", "b_torch", 1)
        result_torch = eval(expr_torch)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())

    @pytest.mark.parametrize("expr", filter(lambda e: "b" in e and e not in [
        "+a",
        "a.minimum(b)",
        "a.maximum(b)",  # not defined for Tensors
        "a[idx]",  # separate
    ], EXPRESSIONS))
    def test_b_array_diff_shape_no_grad(self, expr, shape):
        if len(shape) < 1:
            return

        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b_data = np.atleast_1d(np.random.randn(*shape[1:]))
        b = b_data.copy()
        b_torch = torch.tensor(b_data.copy())

        result = eval(expr)
        result.sum().backward()

        expr_torch = expr.replace("a", "a_torch", 1).replace("b", "b_torch", 1)
        result_torch = eval(expr_torch)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())

    @pytest.mark.parametrize("expr", filter(lambda e: "b" in e and e not in [
        "+a",
        "a.minimum(b)",
        "a.maximum(b)",  # not defined for Tensors
        "a[idx]",  # separate
        "a%b",  # grad for other not defined
    ], EXPRESSIONS))
    def test_b_array_diff_shape_with_grad(self, expr, shape):
        if len(shape) < 1:
            return

        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b_data = np.atleast_1d(np.random.randn(*shape[1:]))
        b = Variable(b_data.copy())
        b_torch = torch.tensor(b_data.copy())
        b_torch.requires_grad = True

        result = eval(expr)
        result.sum().backward()

        expr_torch = expr.replace("a", "a_torch", 1).replace("b", "b_torch", 1)
        result_torch = eval(expr_torch)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())


class TestIndexing:
    @pytest.mark.parametrize("idx", [
        np.s_[0],
        np.s_[3],
        np.s_[:3],
        np.s_[3:],
        np.s_[:0],
        np.s_[0:],
        np.s_[-1:],
        np.s_[:-1],
        np.s_[-4:],
        np.s_[:-8],
        np.s_[[0, 0, 0]],
        np.s_[None],
        np.s_[[1, 2, 3], [0, 4]],
        np.s_[None, ..., None],
        np.s_[[0, 0, 0], [1, 1]],
        np.s_[None, ..., :3, -4, [0, 0]],
    ])
    def test(self, idx, shape):
        np.random.seed(42)
        x_data = np.random.randn(*shape)
        x = Variable(x_data.copy())
        x_torch = torch.tensor(x_data.copy())
        x_torch.requires_grad = True

        result = None
        result_torch = None
        try:
            result = x[idx]
        except IndexError:
            with pytest.raises(IndexError):
                result_torch = x_torch[idx]

            # fine idx is problematic we should skip
            return

        # if this throws IndexError here then there's problem with ours
        # because it didn't
        result_torch = x_torch[idx]

        assert result is not None
        assert result_torch is not None

        result.sum().backward()
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(x.grad, x_torch.grad.numpy())


@pytest.mark.parametrize("func", ["minimum", "maximum"])
class TestMinimumMaximum:
    def test_same(self, func, shape):
        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b_data = np.random.randn(*shape)
        b = Variable(b_data.copy())
        b_torch = torch.tensor(b_data.copy())
        b_torch.requires_grad = True

        result = getattr(a, func)(b)
        result.sum().backward()

        result_torch = getattr(torch, func)(a_torch, b_torch)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())

    def test_random(self, func, shape):
        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b_data = a_data
        b = Variable(b_data.copy())
        b_torch = torch.tensor(b_data.copy())
        b_torch.requires_grad = True

        result = getattr(a, func)(b)
        result.sum().backward()

        result_torch = getattr(torch, func)(a_torch, b_torch)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())


class TestMatMul:
    def test(self, shape):
        if len(shape) > 1:
            # swap last two
            b_shape = (*shape[:-2], shape[-1], shape[-2])
        else:
            b_shape = shape

        np.random.seed(42)
        a_data = np.random.randn(*shape)
        a = Variable(a_data.copy())
        a_torch = torch.tensor(a_data.copy())
        a_torch.requires_grad = True

        b_data = np.random.randn(*b_shape)
        b = Variable(b_data.copy())
        b_torch = torch.tensor(b_data.copy())
        b_torch.requires_grad = True

        result = a @ b
        result.sum().backward()

        result_torch = a_torch @ b_torch
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())


class TestTranspose:
    @pytest.fixture(scope="class", autouse=True, params=range(42, 42 * 2))
    def params(self, shape, request):
        random.seed(request.param)
        n = len(shape)

        source = random.choice(range(-n, n))
        destination = random.choice(range(-n, n))
        return shape, source, destination

    def test(self, params):
        shape, source, destination = params
        x_data = np.ones(shape)
        x = Variable(x_data.copy())
        x_torch = torch.tensor(x_data.copy())
        x_torch.requires_grad = True


        result = x.transpose(source, destination)
        result.sum().backward()

        result_torch = x_torch.transpose(source, destination)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(x.grad, x_torch.grad.numpy())

    def test_t(self, shape):
        if len(shape) > 2:
            return  # corresponding assert will trigger no need

        x_data = np.ones(shape)
        x = Variable(x_data.copy())
        x_torch = torch.tensor(x_data.copy())
        x_torch.requires_grad = True

        result = x.t()
        result.sum().backward()

        result_torch = x_torch.t()
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(x.grad, x_torch.grad.numpy())

class TestReshape:
    def test_complete_reduction(self, shape):
        x_data = np.ones(shape)
        x = Variable(x_data)
        x_torch = torch.from_numpy(x_data)
        x_torch.requires_grad = True

        result = x.reshape(-1)
        result.sum().backward()

        result_torch = x_torch.reshape(-1)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(x.grad, x_torch.grad.numpy())


    def test_complete_redution_and_backwards(self, shape):
        x_data = np.ones(shape)
        x = Variable(x_data)
        x_torch = torch.from_numpy(x_data)
        x_torch.requires_grad = True

        result = x.reshape(-1).reshape(shape)
        result.sum().backward()

        result_torch = x_torch.reshape(-1).reshape(shape)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(x.grad, x_torch.grad.numpy())

