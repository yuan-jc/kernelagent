# Known-wrong candidate: transposes one operand; shapes still match on the
# square problem, so only a real correctness check can reject it.
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B.transpose(0, 1))
