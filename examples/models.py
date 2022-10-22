"""
Implements models using the tiny `simple_autograd.nn` library.
"""
import numpy as np

import simple_autograd.nn as nn
import simple_autograd.nn.functional as F

from simple_autograd import Variable


__all__ = ["MLP", "CNN", "ViT"]


class MLP(nn.Module):
    """
    A multi-layer perceptron.
    """
    def __init__(self, sizes, flatten_first=False, norm_layer=None):
        super().__init__()
        self.sizes = sizes
        self.flatten_first = flatten_first

        layers = []
        if flatten_first:
            layers = [nn.Flatten()]

        prev = sizes[0]
        for h in sizes[1:-1]:
            layers += [nn.Linear(prev, h)]
            if norm_layer:
                layers += [norm_layer(h)]
            layers += [nn.ReLU()]
            prev = h

        layers += [nn.Linear(prev, sizes[-1])]

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
        layers += [nn.GlobalAvgPool2d(), nn.Linear(second_layer_chs, output_classes)]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class ViTBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        # multi-head self attention
        self.mhsa = nn.MultiHeadSelfAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        hidden_size = int(mlp_ratio * emb_dim)
        self.mlp = MLP(sizes=[emb_dim, hidden_size, emb_dim])
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, single_query=False):
        res1 = x
        if single_query:
            res1 = x[:, :1]
        x = res1 + self.mhsa(self.norm1(x), single_query=single_query)
        x = x + self.mlp(self.norm2(x))

        return x


class ViT(nn.Module):
    """
    ViT based on https://www.youtube.com/watch?v=ovB0ddFtzzA
    """
    def __init__(self, img_size, in_channels, patch_size, num_classes, emb_dim, num_heads, num_blocks):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = (patch_size, patch_size)
        self.num_classes = num_classes

        num_patch_pixels = patch_size * patch_size
        self.patch_linear_proj = nn.Linear(num_patch_pixels * in_channels, emb_dim)

        self.cls_token = Variable(np.zeros((1, 1, emb_dim)))
        self.pos_embed = Variable(np.zeros((1, 1 + (img_size // patch_size) ** 2, emb_dim)))

        blocks = []
        for _ in range(num_blocks - 1):
            blocks += [ViTBlock(emb_dim, num_heads)]

        self.blocks = nn.Sequential(*blocks)
        self.last_block = ViTBlock(emb_dim, num_heads)

        self.final_linear_proj = nn.Linear(emb_dim, num_classes)

    def get_patch_embeddings(self, x):
        # x : N, C, H, W
        x = F.get_2D_patches(x, self.patch_size)  # N, C, oH, oW, pH, pW

        N, C, oH, oW, pH, pW = x.shape
        x = x.swapaxes(1, 3)  # N, oW, oH, C, pH, pW

        x = x.reshape((N, oH * oW, C * pH * pW))  # N, L, E'

        return x

    def forward(self, x):
        # x : N, C, H, W
        x = self.get_patch_embeddings(x)  # N, L, E'
        x = self.patch_linear_proj(x)  # N, L, E
        cls_token = self.cls_token[x.shape[0] * [0]]  # expand to N
        x = cls_token.concat(x, axis=1)

        x = x + self.pos_embed

        # transformer
        for block in self.blocks:
            x = block(x)
        x = self.last_block(x, single_query=True)  # N, 1, E

        N, _, _ = x.shape
        x = self.final_linear_proj(x).reshape((N, self.num_classes))

        return x
