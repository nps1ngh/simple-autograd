"""
Multi-head **self** attention layer.
As the ViT doesn't require a more generalized attention layer,
we only implement this.
"""
from .base import Module
from .linear import Linear


class MultiHeadSelfAttention(Module):
    def __init__(self, emb_dim, num_heads, bias=True):
        assert emb_dim % num_heads == 0, "emb_dim should be divisible by num_head!"
        super().__init__()
        self.dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = Linear(emb_dim, 3 * emb_dim, bias=bias)
        self.proj = Linear(emb_dim, emb_dim)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}"
        )

    def forward(self, x, single_query=False):
        N, L, E = x.shape
        qkv = self.qkv(x)  # (N, L, 3E)
        qkv = qkv.reshape((N, L, 3, self.num_heads, self.head_dim))  # N, L, 3, Nh, H
        qkv = (
            qkv  # N, L, 3, Nh, H
            .swapaxes(0, 2)  # 3, L, N, Nh, H
            .swapaxes(1, 3)  # 3, Nh, N, L, H
            .swapaxes(1, 2)  # 3, N, Nh, L, H
        )

        q, k, v = qkv  # N, Nh, L, H
        if single_query:
            q = q[:, :, :1, :]  # N, Nh, 1, H
        k_t = k.swapaxes(2, 3)  # N, Nh, H, L
        similarities = (q @ k_t) * self.scale  # N, Nh, L|1, L

        attn_weights = similarities.softmax(3)  # N, Nh, L|1, L
        result = attn_weights @ v  # N, Nh, L|1, H
        result = (
            result
            .swapaxes(1, 2)  # N, L|1, Nh, H
            .reshape((N, 1 if single_query else L, E))
        )

        result = self.proj(result)  # N, L|1, E

        return result







