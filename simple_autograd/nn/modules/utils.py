"""
Contains utility modules and functions.
"""
from .base import Module


def count_parameters(m: Module, requires_grad: bool = True):
    """
    Counts the number of parameters in the given module `m`.
    Parameters
    ----------
    m : Module
        The module, whose parameters need to be counted.
    requires_grad : bool
        Whether to only count parameters requiring grad.
        Default: True

    Returns
    -------
    int
        Number of total parameters in the module.
    """
    return sum(p.size for p in m.parameters() if p.requires_grad == requires_grad)


class Identity(Module):
    def forward(self, input):
        return input


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()

        for i, m in enumerate(modules):
            setattr(self, f"{i}", m)

    def __iter__(self):
        return iter(self._submodules.values())

    def __getitem__(self, item) -> Module:
        return self._submodules[f"{item}"]

    def forward(self, input):
        for m in self:
            input = m(input)
        return input


class Flatten(Module):
    def forward(self, input):
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)


class Exp(Module):
    # never seen this being used anywhere but here you go
    def forward(self, input):
        return input.exp()


class Sigmoid(Module):
    def forward(self, input):
        return input.sigmoid()


class Softmax(Module):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def forward(self, input):
        if self.axis is None:
            axis = 0
            input = input.reshape(-1)
        else:
            axis = self.axis

        return input.softmax(axis)
