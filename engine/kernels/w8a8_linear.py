"""Experimental W8A8 decode linear using INT8 tensor-core dot products."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a8_kernel(x_ptr, w_ptr, ws_ptr, out_ptr, M, N, K,
                 sxm, sxk, swn, swk, som, son,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    om, on, ok = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN), tl.arange(0, BK)
    wscale = tl.load(ws_ptr + on, mask=on < N, other=0.0).to(tl.float32)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, K, BK):
        x = tl.load(x_ptr + om[:, None] * sxm + (start + ok[None, :]) * sxk,
                    mask=(om[:, None] < M) & (start + ok[None, :] < K), other=0.0).to(tl.float32)
        xscale = tl.maximum(tl.max(tl.abs(x), axis=1) / 127.0, 1e-8)
        xq = tl.where(x / xscale[:, None] >= 0, x / xscale[:, None] + 0.5, x / xscale[:, None] - 0.5)
        xq = tl.maximum(tl.minimum(xq, 127.0), -127.0).to(tl.int8)
        wq = tl.load(w_ptr + on[None, :] * swn + (start + ok[:, None]) * swk,
                     mask=(on[None, :] < N) & (start + ok[:, None] < K), other=0)
        acc += tl.dot(xq, wq).to(tl.float32) * (xscale[:, None] * wscale[None, :])
    tl.store(out_ptr + om[:, None] * som + on[None, :] * son, acc.to(out_ptr.dtype.element_ty),
             mask=(om[:, None] < M) & (on[None, :] < N))


def w8a8_linear(inputs: torch.Tensor, qweight: torch.Tensor, weight_scales: torch.Tensor) -> torch.Tensor:
    """Per-K-tile activation quantization plus INT8 tensor-core dot accumulation."""
    if inputs.ndim != 2 or qweight.dtype is not torch.int8 or qweight.ndim != 2:
        raise ValueError("inputs must be 2D and qweight must be 2D INT8")
    M, K = inputs.shape; N, wk = qweight.shape
    if wk != K or weight_scales.shape != (N,) or K % 32:
        raise ValueError("invalid W8A8 geometry; K must be divisible by 32")
    output = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype)
    _w8a8_kernel[(triton.cdiv(M, 16), triton.cdiv(N, 128))](
        inputs, qweight, weight_scales, output, M, N, K,
        inputs.stride(0), inputs.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1),
        BM=16, BN=128, BK=32, num_warps=4,
    )
    return output
