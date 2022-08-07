"""
Tests for the `nn` sub-library.
We do not test everything since almost all the correctness checking is
already done via `test_variable.py`.
Also, the point of this was implement autograd engine and not a neural network
framework. (The `nn` sub-library is only used to show what it's capable of.)
"""
import random

import pytest
import torch
import numpy as np
import torch.nn as nn_torch
import torch.nn.functional as F_torch

import simple_autograd.nn as nn
import simple_autograd.nn.functional as F
from simple_autograd import Variable


NUM = 10  # this will generate 3 * NUM * NUM many tests
MAX = 1000


def BATCH_SIZES():
    random.seed(42)
    return [random.randint(1, MAX) for _ in range(NUM)]


def DIMENSIONS():
    random.seed(42 * 2)
    return [random.randint(1, MAX) for _ in range(NUM)]


# The CE loss is the only untested thing
class TestCELoss:
    @pytest.mark.parametrize("batch_size", BATCH_SIZES())
    @pytest.mark.parametrize("dimensions", DIMENSIONS())
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test(self, reduction, batch_size, dimensions):
        x_data = np.ones((batch_size, dimensions))
        x = Variable(x_data.copy())
        x_torch = torch.tensor(x_data.copy())
        x_torch.requires_grad = True

        t = np.random.randint(0, dimensions, size=(batch_size,), dtype=np.int64)
        t_torch = torch.tensor(t)

        # note this is our functional not torch's functional
        result = F.cross_entropy(x, t, reduction=reduction)
        result.sum().backward()

        result_torch = F_torch.cross_entropy(x_torch, t_torch, reduction=reduction)
        result_torch.sum().backward()

        np.testing.assert_array_almost_equal(result.data, result_torch.detach().numpy())
        np.testing.assert_array_almost_equal(x.grad, x_torch.grad.numpy())


class TestSimple:
    def test(self):
        np.random.seed(42)

        IN_SIZE = 28 * 28
        BATCH_SIZE = 64

        t = np.random.randint(0, 10, size=(BATCH_SIZE,), dtype=np.int64)

        # this is our nn
        model = nn.Sequential(
            nn.Linear(IN_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
        )

        x = np.ones((BATCH_SIZE, IN_SIZE))
        result = model(x)
        result = F.cross_entropy(result, t)

        linear_layers_torch = [
            (
                torch.from_numpy(m.weight).requires_grad_(),
                torch.from_numpy(m.bias).requires_grad_(),
            )
            for m in model
            if isinstance(m, nn.Linear)
        ]
        x = torch.ones(BATCH_SIZE, IN_SIZE, dtype=torch.float64)
        x = x @ linear_layers_torch[0][0].t() + linear_layers_torch[0][1]
        x = torch.relu(x)
        x = x @ linear_layers_torch[1][0].t() + linear_layers_torch[1][1]
        x = torch.relu(x)
        x = x @ linear_layers_torch[2][0].t() + linear_layers_torch[2][1]
        x = torch.relu(x)
        result_torch = F_torch.cross_entropy(x, torch.tensor(t))

        np.testing.assert_allclose(
            result.view(np.ndarray),
            result_torch.detach().numpy(),
        )

        result.backward()
        result_torch.backward()

        for i in range(3):
            np.testing.assert_allclose(
                model[2 * i].weight.grad.view(np.ndarray),
                linear_layers_torch[i][0].grad.numpy(),
            )
            np.testing.assert_allclose(
                model[2 * i].bias.grad.view(np.ndarray),
                linear_layers_torch[i][1].grad.numpy(),
            )


class TestConvolution:
    def test(self):
        import time
        torch.manual_seed(42)
        N = 32
        C = 3
        R = 31
        K = 7
        O = 6
        x_torch = torch.randn(N, C, R, R, requires_grad=True, dtype=torch.float64)
        w_torch = torch.randn(O, C, K, K, requires_grad=True, dtype=torch.float64)

        x = Variable(x_torch.detach().numpy())
        w = Variable(w_torch.detach().numpy())

        START = time.perf_counter()
        out_torch = torch.conv2d(x_torch, w_torch)
        END = time.perf_counter()
        print()
        print(f"Torch's took time: {END - START}")
        print()

        START = time.perf_counter()
        out = nn.functional.conv2d(x, w)
        END = time.perf_counter()
        print()
        print(f"Ours took time: {END - START}")
        print()

        np.testing.assert_allclose(out.view(np.ndarray), out_torch.detach().numpy())

        # now do backward pass
        START = time.perf_counter()
        out_torch.sum().backward()
        END = time.perf_counter()
        print()
        print(f"Torch's backward took time: {END - START}")
        print()

        START = time.perf_counter()
        out.sum().backward()
        END = time.perf_counter()
        print()
        print(f"Our backward took time: {END - START}")
        print()

        np.testing.assert_allclose(x.grad, x_torch.grad.numpy())
        np.testing.assert_allclose(w.grad, w_torch.grad.numpy())




















