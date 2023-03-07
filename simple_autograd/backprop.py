from __future__ import annotations

import graphlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from . import variable


def _get_order(var: variable.Variable) -> list[variable.Variable]:
    """
    Returns the top sort in reverse order for back-propagating through
    the computational graph.

    Parameters
    ----------
    var : variable.Variable
        The (root/end) variable from where to start.

    Returns
    -------
    list[variable.Variable]
        Variables ordered according to when to call the corresponding
        backward method.
    """
    assert var.grad_fn is not None, "grad_fn should not be None!"
    ts = graphlib.TopologicalSorter()

    stack: list[variable.Variable] = [var]
    while len(stack) > 0:
        next_var = stack.pop()
        children = next_var.grad_fn.get_inputs()
        stack.extend(children)

        ts.add(next_var, *children)

    return list(reversed(list(ts.static_order())))


def backward(var: variable.Variable):
    """
    Start the backward pass from the given variable.

    Parameters
    ----------
    var : variable.Variable
    The variable to start from. Should be a scalar array.

    """
    assert var.requires_grad, "Variable does not require grad!"
    # call item to make sure it's a scalar
    _ = var.item()

    # get topsort
    order = _get_order(var)
    assert (
        order[0] is var
    ), "first node should be the one on which .backward() was called!"

    var.grad = np.ones_like(var.data)
    for v in order:
        assert not v.requires_grad or v.grad is not None, f"Gradient was None of {v}"
        v.grad_fn.backprop(out_grad=v.grad)

        # clear out if not needed anymore
        if not v.retains_grad:
            v.grad = None
