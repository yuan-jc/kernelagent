# Example candidate for the rmsnorm_fp32 smoke run (not part of the problem
# contract): torch's fused F.rms_norm behind the same entry symbol.
import torch
import torch.nn.functional as F


def run(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, x.shape[-1:], None, 1.0e-6)
