"""
Implements models using the tiny `simple_autograd.nn` library.
"""
import simple_autograd.nn as nn


__all__ = ["MLP", "CNN"]


class MLP(nn.Module):
    """
    A multi-layer perceptron.
    """
    def __init__(self, sizes, flatten_first=False):
        super().__init__()
        self.sizes = sizes
        self.flatten_first = flatten_first

        layers = []
        if flatten_first:
            layers = [nn.Flatten()]

        prev = sizes[0]
        for h in sizes[1:]:
            layers += [nn.Linear(prev, h)]

            layers += [nn.ReLU()]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CNN(nn.Module):
    """
    A very basic 2-layer CNN model.
    """
    def __init__(self, input_channels, hidden_channels, kernel_sizes, max_pool_ks, output_classes, norm_layer=None):
        assert len(hidden_channels) == 2, "Expected a length 2 sequence of hidden channels!"
        assert len(kernel_sizes) == 2, "Expected a length 2 sequence of kernel sizes!"
        assert len(max_pool_ks) == 2, "Expected a length 2 sequence of max pool kernel sizes!"
        super().__init__()

        first_layer_chs, second_layer_chs = hidden_channels
        first_layer_ks, second_layer_ks = kernel_sizes
        first_layer_mpks, second_layer_mpks = max_pool_ks

        layers = []

        # first layer
        layers += [nn.Conv2d(input_channels, first_layer_chs, first_layer_ks)]
        if norm_layer:
            layers += [norm_layer(first_layer_chs)]
        layers += [nn.ReLU()]

        # first pooling
        layers += [nn.MaxPool2d(first_layer_mpks)]

        # second layer
        layers += [nn.Conv2d(first_layer_chs, second_layer_chs, second_layer_ks)]
        if norm_layer:
            layers += [norm_layer(second_layer_chs)]
        layers += [nn.ReLU()]

        # 2nd pool
        layers += [nn.MaxPool2d(second_layer_mpks)]

        # final layer
        layers += [nn.Global1x1AvgPool2d(), nn.Linear(second_layer_chs, output_classes)]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

