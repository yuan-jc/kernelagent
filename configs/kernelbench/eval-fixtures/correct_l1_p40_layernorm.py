# Known-correct candidate: mirrors the pinned reference structure exactly,
# so the upstream set_seed weight-sync produces identical parameters.
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        return self.ln(x)
