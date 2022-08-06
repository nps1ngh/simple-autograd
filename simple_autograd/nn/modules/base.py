"""
Contains the base class for all modules.
"""
import abc
import itertools
from typing import Iterator, Any

from ...variable import Variable


def _addindent(s_, numSpaces):
    """
    Taken from torch.nn.modules.module
    """
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module(abc.ABC):
    def __init__(self):
        self._params: list[Variable] = []
        self._submodules: list[Module] = []
        self._submodule_names: list[str] = []

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Variable]:
        return itertools.chain(
            self._params, *(m.parameters() for m in self._submodules)
        )

    def __setattr__(self, key, value):
        if isinstance(value, Variable) and value.requires_grad:
            self._params.append(value)
        elif issubclass(type(value), Module):
            self._submodules.append(value)
            self._submodule_names.append(key)

        super().__setattr__(key, value)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        # this is modified from torch.nn.Module.__repr__
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in zip(self._submodule_names, self._submodules):
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = type(self).__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
