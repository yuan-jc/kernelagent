# user-bench reference: runs only inside a worker process; the file bytes are
# frozen by the suite manifest (sha256) and re-verified before every run.
import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused add + ReLU; the reference output is the correctness ground truth."""
    return torch.relu(torch.add(a, b))
