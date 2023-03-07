"""
Contains loss implementations.
"""
from .. import functional as F
from .base import Module


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        assert reduction in ["none", "mean", "sum"]

        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, reduction=self.reduction)
