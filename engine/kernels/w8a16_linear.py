"""Experimental W8A16 linear kernel for decode-sized Qwen projections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_linear_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_on, stride_om,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (start_k + offs_k[None, :]) * stride_xk,
            mask=(offs_m[:, None] < M) & (start_k + offs_k[None, :] < K), other=0.0,
        )
        weight = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + (start_k + offs_k[:, None]) * stride_wk,
            mask=(offs_n[None, :] < N) & (start_k + offs_k[:, None] < K), other=0,
        ).to(tl.float16)
        scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float16)
        accumulator += tl.dot(x, weight * scales[None, :])
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             accumulator.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack `[out_features, in_features]` FP16 weights into INT8 plus FP16 scales."""
    scales = weight.float().abs().amax(dim=1).clamp_min(1e-8).div(127).to(torch.float16)
    quantized = torch.round(weight.float() / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scales.contiguous()


def w8a16_linear(inputs: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                 bias: torch.Tensor | None = None) -> torch.Tensor:
    """W8A16 linear for 2D decode activations; dequantization happens in the kernel."""
    if inputs.ndim != 2 or qweight.ndim != 2 or qweight.dtype is not torch.int8:
        raise ValueError("inputs must be 2D and qweight must be 2D INT8")
    M, K = inputs.shape
    N, weight_k = qweight.shape
    if weight_k != K or scales.shape != (N,) or (bias is not None and bias.shape != (N,)):
        raise ValueError("incompatible W8A16 linear geometry")
    if K % 32:
        raise ValueError("W8A16 kernel requires input width divisible by 32")
    output = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)
    block_m, block_n, block_k = 16, 64, 32
    _w8a16_linear_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](
        inputs, qweight, scales, bias if bias is not None else output, output,
        M, N, K, inputs.stride(0), inputs.stride(1), qweight.stride(0), qweight.stride(1),
        output.stride(1), output.stride(0), HAS_BIAS=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
    )
    return output
