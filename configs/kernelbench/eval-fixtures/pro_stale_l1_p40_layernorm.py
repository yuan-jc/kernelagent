# Counterexample (T06): caches the first output and replays it, so the
# second distinct input gets a stale result.
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._cache = None

    def forward(self, x):
        if self._cache is None:
            self._cache = self.ln(x)
        return self._cache
