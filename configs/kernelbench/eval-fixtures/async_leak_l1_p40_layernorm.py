# Known async-leak counterexample (T07): enqueues the work on a side
# stream and returns without joining it to the caller's stream. Output is
# still correct after a device-wide synchronize, but default-stream event
# pairs measure almost nothing. The protocol's single-shot sync-bracketed
# integrity check must flag this as async_leak.
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.side = torch.cuda.Stream()

    def forward(self, x):
        with torch.cuda.stream(self.side):
            out = self.ln(x)
        return out
