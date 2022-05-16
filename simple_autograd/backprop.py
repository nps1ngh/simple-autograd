from __future__ import annotations
import graphlib
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import variable


def _get_order(var: variable.Variable) -> list[variable.Variable]:
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
    assert var.requires_grad, "Variable does not require grad!"
    # call item to make sure it's a scalar
    _ = var.item()

    # get topsort
    order = _get_order(var)
    assert (
        order[0] == var
    ), "first node should be the one on which .backward() was called!"

    var.grad = np.ones_like(var.data)
    for v in order:
        assert not v.requires_grad or v.grad is not None, f"Gradient was None of {v}"
        v.grad_fn.backprop(out_grad=v.grad)

        # clear out if not needed anymore
        if not v.retain_grad:
            v.grad = None
