# Example candidate for the add_relu_fp32 smoke run (not part of the problem
# contract): a flat Triton elementwise kernel behind the same entry symbol.
import torch
import triton
import triton.language as tl


@triton.jit
def _add_relu_kernel(a_ptr, b_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    y = tl.maximum(a + b, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(a)
    n_elements = a.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _add_relu_kernel[grid](a, b, y, n_elements, BLOCK=1024)
    return y
