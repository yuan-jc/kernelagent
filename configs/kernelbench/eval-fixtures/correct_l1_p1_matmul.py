# Known-correct candidate: identical math to the pinned reference
# (KernelBench/level1/1_Square_matrix_multiplication_.py).
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)
