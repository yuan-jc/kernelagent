# Compile-baseline candidate: same reference math executed through
# torch.compile (exercise of the compile path through the upstream
# evaluator; the formal timing protocol is T07).
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._compiled = None

    def forward(self, x):
        if self._compiled is None:
            self._compiled = torch.compile(self._normalized)
        return self._compiled(x)

    def _normalized(self, x):
        return self.ln(x)
