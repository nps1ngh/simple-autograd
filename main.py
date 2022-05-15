import numpy as np

import simple_autograd.variable as variable


def main1():
    x = np.ones(3) * 3
    print(f"{x=}")
    x = variable.Variable(x)
    print(f"{x=}")

    y = np.ones((4, 3)) - x
    print(f"{y=}")

    z = (y * np.array([2, 2, 2], dtype=float)).sum()
    print(f"{z=}")

    z.backward()
    print(f"{x.grad=}")
    print(f"{y.grad=}")
    print(f"{z.grad=}")


def main2():
    x = np.arange(3, 6).reshape(-1, 1) * np.arange(1, 4)
    x = np.asarray(x, float)
    x = variable.Variable(x, requires_grad=True)
    print(f"{x=}")

    y = np.array([0, 1, 0]) * x
    print(y)

    z = y.sum(1)
    print(z)

    u = np.array([0, 1, 0]) * z
    print(u)
    t = u.sum()
    print(t)
    t.backward()

    print(f"{x.grad=}")
    print("Now torch")
    import torch
    x = torch.arange(3, 6).reshape(-1, 1) * torch.arange(1, 4)
    x = x.float()
    x.requires_grad = True
    print(f"{x=}")

    y = torch.tensor([0, 1, 0]) * x
    print(y)

    z = y.sum(1)
    print(z)

    u = torch.tensor([0, 1, 0]) * z
    print(u)
    t = u.sum()
    print(t)
    t.backward()

    print(f"{x.grad=}")

if __name__ == '__main__':
    main2()
