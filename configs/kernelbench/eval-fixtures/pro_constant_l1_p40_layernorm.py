# Counterexample (T06): returns a constant regardless of input.
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        return torch.zeros_like(x)
