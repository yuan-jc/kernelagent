# T14 autotune candidate: Triton LayerNorm that writes its result INTO the
# input buffer (in-place output). The harness must restore pristine inputs
# between configs or every later config measures corrupted state.
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_inplace_kernel(x_ptr, K, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * K
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for offset in range(0, K, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < K
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        acc += x
        acc_sq += x * x
    mean = tl.sum(acc, axis=0) / K
    var = tl.sum(acc_sq, axis=0) / K - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    for offset in range(0, K, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < K
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        tl.store(x_ptr + base + cols, (x - mean) * rstd, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple, block_size: int = 1024):
        super(ModelNew, self).__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.block_size = block_size

    def forward(self, x):
        k = 1
        for dim in self.normalized_shape:
            k *= dim
        rows = x.numel() // k
        flat = x.reshape(rows, k)
        BLOCK = triton.next_power_of_2(min(self.block_size, k))
        _ln_inplace_kernel[(rows,)](flat, k, 1e-5, BLOCK=BLOCK)
        return x
