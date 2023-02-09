import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """
    This is a very simple implementation of SGD.
    """

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        self.steps = 0

        if self.momentum > 0:
            self.vel = [np.zeros(p.shape) for p in self.parameters]
        else:
            self.vel = None

    def step(self) -> None:
        self.steps += 1  # increment step counter

        for i, p in enumerate(self.parameters):
            g = p.grad if p.grad is not None else np.zeros(p.shape)

            if self.weight_decay != 0:
                g = g + self.weight_decay * p.view(np.ndarray)

            if self.momentum > 0:
                if self.steps > 1:
                    self.vel[i] = self.momentum * self.vel[i] + (1 - self.dampening) * g
                else:
                    self.vel[i] = g

                if self.nesterov:
                    g = g + self.momentum * self.vel[i]
                else:
                    g = self.vel[i]

            step = self.lr * g
            p = p.view(np.ndarray)
            if self.maximize:
                np.add(p, step, out=p)
            else:
                np.subtract(p, step, out=p)

    def state_dict(self):
        if self.vel is None:
            return super().state_dict()
        else:
            return {
                str(i): p
                for i, p in enumerate(self.vel)
            }

    def load_state_dict(self, state_dict):
        if self.vel is None:
            return
        
        size = 0 if self.vel is None else len(self.vel)
        assert len(state_dict) == size

        self.vel = [state_dict[str(i)] for i in range(size)]
