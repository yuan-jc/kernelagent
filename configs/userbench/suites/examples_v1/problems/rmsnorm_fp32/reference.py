# user-bench reference: runs only inside a worker process; the file bytes are
# frozen by the suite manifest (sha256) and re-verified before every run.
import torch


def run(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm (no affine weights); the reference output is the ground truth."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + 1.0e-6)
