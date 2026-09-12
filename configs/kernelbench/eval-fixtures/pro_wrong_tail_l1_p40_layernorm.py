# Counterexample (T06): correct everywhere except the tail of the flattened
# output, which is zeroed.
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        out = self.ln(x)
        flat = out.reshape(-1)
        flat[-64:] = 0.0
        return out
