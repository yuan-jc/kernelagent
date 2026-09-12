# Counterexample (T06): mutates its input in place. LayerNorm output is
# scale-invariant, so a pure output comparison would still pass - only the
# input-mutation check catches this.
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        x.mul_(2.0)
        return self.ln(x)
