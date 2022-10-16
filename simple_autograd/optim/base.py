"""
Base optimizer class.
Does not support any parameter groups.
"""
import abc

from ..variable import Variable


class Optimizer(abc.ABC):
    def __init__(self, params):
        self.parameters: list[Variable] = [p for p in params if p.requires_grad]

        assert len(self.parameters) > 0, "Got 0 parameters (requiring grad)!"

    @abc.abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()
