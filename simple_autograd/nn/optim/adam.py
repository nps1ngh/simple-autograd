import numpy as np

from .base import Optimizer


class Adam(Optimizer):
    """
    This is a very simple implementation of Adam.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        *,
        maximize=False,
    ):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.steps = 0

        self.m = [np.zeros(p.shape) for p in self.parameters]
        self.v = [np.zeros(p.shape) for p in self.parameters]

    def step(self) -> None:
        self.steps += 1  # increment step counter

        beta1, beta2 = self.betas
        for i, p in enumerate(self.parameters):
            g = p.grad if p.grad is not None else np.zeros(p.shape)
            if self.maximize:
                g = -g

            if self.weight_decay != 0:
                g = g + self.weight_decay * p.view(np.ndarray)

            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g**2)
            m = self.m[i] / (1 - beta1**self.steps)
            v = self.v[i] / (1 - beta2**self.steps)

            step = self.lr * m / (np.sqrt(v) + self.eps)

            p = p.view(np.ndarray)
            np.subtract(p, step, out=p)

    def state_dict(self):
        m_sd = {f"m.{i}": p for i, p in enumerate(self.m)}
        v_sd = {f"v.{i}": p for i, p in enumerate(self.v)}
        m_sd.update(v_sd)
        assert len(m_sd) == 2 * len(self.m)
        return m_sd

    def load_state_dict(self, state_dict):
        assert len(state_dict) == 2 * len(self.m)

        self.m = [state_dict[f"m.{i}"] for i in range(len(self.m))]
        self.v = [state_dict[f"v.{i}"] for i in range(len(self.v))]
