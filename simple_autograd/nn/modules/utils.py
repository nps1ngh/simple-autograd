"""
Contains utility modules.
"""
from .base import Module


class Identity(Module):
    def forward(self, input):
        return input


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()

        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def __iter__(self):
        return self._submodules.values()

    def __getitem__(self, item) -> Module:
        return self._submodules[item]

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
