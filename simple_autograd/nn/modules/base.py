"""
Contains the base class for all modules.
"""
import abc
import itertools
from typing import Iterator, Any, Union

import numpy as np

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
        self._params: dict[str, Variable] = {}
        self._buffers: dict[str, Union[np.ndarray, Variable]] = {}
        self._submodules: dict[str, Module] = {}

        self.training: bool = True
        """ Whether the module is in training mode. """

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Variable]:
        return itertools.chain(
            self._params.values(), *(m.parameters() for m in self._submodules.values())
        )

    def __setattr__(self, key, value):
        if isinstance(value, Variable) and value.requires_grad:
            self._params[key] = value
        elif isinstance(value, np.ndarray):
            self._buffers[key] = value
        elif issubclass(type(value), Module):
            self._submodules[key] = value

        super().__setattr__(key, value)

    def train(self, mode: bool = True) -> None:
        if self.training != mode:
            self.training = mode
            for m in self._submodules.values():
                m.train()

    def eval(self) -> None:
        self.train(False)

    def state_dict(self) -> dict[str, Union[Variable, np.ndarray]]:
        """
        Creates the state dict of this Module.
        Meant to be saved using NumPy's np.savez function.
        **In particular, the ** unary operator needs to be used.**

        Example:
        ---
        ```python
        model = ...
        out_file = ...
        state_dict = model.state_dict()
        np.savez(out_file, **state_dict)
        ```

        :return: The state dict of the module.
        :rtype: dict[str, Union[Variable, np.ndarray]]
        """
        result = {}

        # update with buffers
        result.update(self._params)
        result.update(self._buffers)

        # for children attach prefix
        result.update(
            (f"{child_name}.{child_key}", child_value)
            for child_name, child in self._submodules.items()
            for child_key, child_value in child.state_dict().items()
        )

        return result

    def load_state_dict(self, state_dict: dict[str, Union[Variable, np.ndarray]], prefix: str = "") -> None:
        """
        Loads the given state dict.
        Note that this is a very This is a very unsafe version of `load_state_dict`.

        :param state_dict: The state dict of the module.
        :type state_dict: dict[str, Union[Variable, np.ndarray]]
        :param prefix: the prefix for this (sub) module. Default is ""
        :type prefix: str
        """
        for key in self._params:
            setattr(self, key, Variable(state_dict[prefix + key]))

        for key in self._buffers:
            setattr(self, key, state_dict[prefix + key])

        for key, value in self._submodules.items():
            value.load_state_dict(state_dict, prefix=prefix + key + ".")

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
        for key, module in self._submodules.items():
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
