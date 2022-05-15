import numpy as np

import simple_autograd.variable as variable


def main():
    x = np.ones(3) * 3
    print(f"{x=}")
    x = variable.Variable(x)
    print(f"{x=}")

    y = np.ones((4, 3)) - x
    print(f"{y=}")

    z = (y * np.array([2, 2, 2], dtype=float)) .sum()
    print(f"{z=}")

    z.backward()
    print(f"{x.grad=}")
    print(f"{y.grad=}")
    print(f"{z.grad=}")


if __name__ == '__main__':
    main()
