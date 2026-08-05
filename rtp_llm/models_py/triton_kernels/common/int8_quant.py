"""Dynamic symmetric per-token INT8 activation quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    n_cols,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale = absmax / 127.0
    x_q = tl.clamp(x / scale, -128.0, 127.0)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale)


def per_token_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension independently for every input row.

    Returns the INT8 values plus one FP32 scale per row, so the scale shape is
    ``(*x.shape[:-1], 1)``. Symmetric, so there is no zero point.
    """
    if x.dim() < 2:
        raise ValueError(f"per-token INT8 quant expects ndim >= 2, got {x.dim()}")
    x_contiguous = x.contiguous()
    x_q = torch.empty_like(x_contiguous, dtype=torch.int8)
    scales = torch.empty(
        (*x_contiguous.shape[:-1], 1),
        device=x_contiguous.device,
        dtype=torch.float32,
    )
    if x_contiguous.numel() == 0:
        return x_q, scales

    rows = x_contiguous.numel() // x_contiguous.shape[-1]
    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    num_warps = 2
    _per_token_quant_int8_kernel[(rows,)](
        x_contiguous,
        x_q,
        scales,
        stride_x=x_contiguous.stride(-2),
        stride_xq=x_q.stride(-2),
        n_cols=n_cols,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, scales
