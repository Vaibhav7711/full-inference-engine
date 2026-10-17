"""Experimental GEMV directly from GEMM-format AWQ 4-bit weights.

This is a projection-only research kernel, not an engine backend. The packed
column order follows the AWQ GEMM format's [0, 4, 1, 5, 2, 6, 3, 7] inverse
ordering; qzeros use the same order. Only group size 128 is measured here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _awq_gemv_kernel(
    x_ptr, qw_ptr, qz_ptr, scales_ptr, out_ptr,
    stride_xb, stride_xk, stride_qwk, stride_qwn,
    stride_qzg, stride_qzn, stride_sg, stride_sn,
    stride_ob, stride_on,
    K: tl.constexpr, N: tl.constexpr, SPLIT_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    tile_n = tl.program_id(1)
    split = tl.program_id(2)
    n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    # Packed nibble position for each logical output channel.
    nibble = (n % 8) // 2 + ((n % 8) % 2) * 4
    shift = nibble * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start_k in range(split * SPLIT_LEN, (split + 1) * SPLIT_LEN, BLOCK_K):
        k = start_k + k_offsets
        x = tl.load(x_ptr + batch * stride_xb + k * stride_xk, mask=k < K, other=0)
        packed = tl.load(
            qw_ptr + k[:, None] * stride_qwk + (n[None, :] // 8) * stride_qwn,
            mask=(k[:, None] < K) & (n[None, :] < N), other=0,
        )
        q = (packed >> shift[None, :]) & 15
        group = k // 128
        packed_zero = tl.load(
            qz_ptr + group[:, None] * stride_qzg + (n[None, :] // 8) * stride_qzn,
            mask=(k[:, None] < K) & (n[None, :] < N), other=0,
        )
        zero = (packed_zero >> shift[None, :]) & 15
        scale = tl.load(
            scales_ptr + group[:, None] * stride_sg + n[None, :] * stride_sn,
            mask=(k[:, None] < K) & (n[None, :] < N), other=0,
        )
        weight = (q - zero).to(tl.float32) * scale.to(tl.float32)
        acc += tl.sum(x[:, None].to(tl.float32) * weight, axis=0)
    tl.store(out_ptr + batch * stride_ob + split * N * stride_on + n * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=n < N)


@triton.jit
def _merge_awq_splits(partial_ptr, out_ptr, N: tl.constexpr, SPLITS: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    batch = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.arange(0, SPLITS)
    values = tl.load(partial_ptr + batch * SPLITS * N + split[:, None] * N + n[None, :],
                     mask=n[None, :] < N, other=0)
    result = tl.sum(values, axis=0)
    tl.store(out_ptr + batch * N + n, result.to(out_ptr.dtype.element_ty), mask=n < N)


def awq_gemv(
    x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
    scales: torch.Tensor, *, block_k: int = 32, block_n: int = 32,
    splits: int = 1,
) -> torch.Tensor:
    """Compute [B,K] @ dequant_AWQ([K,N]) for a small decode batch."""
    if x.ndim != 2 or qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("AWQ GEMV expects rank-2 tensors")
    batch, k = x.shape
    if batch < 1 or k % 128 or qweight.shape[0] != k:
        raise ValueError("input K must match packed weights and be divisible by 128")
    n = scales.shape[1]
    if n % 8 or qweight.shape[1] != n // 8 or qzeros.shape != (k // 128, n // 8) \
            or scales.shape[0] != k // 128:
        raise ValueError("AWQ 4-bit group-128 tensor shapes do not match")
    if x.device.type != "cuda" or any(t.device != x.device for t in (qweight, qzeros, scales)):
        raise ValueError("all tensors must be on the same CUDA device")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("packed AWQ weights and zeros must be int32")
    if block_k not in (16, 32, 64, 128) or block_n not in (8, 16, 32, 64):
        raise ValueError("unsupported tile size")
    if splits < 1 or splits & (splits - 1) or k % splits or (k // splits) % block_k:
        raise ValueError("splits must be a power of two and divide K into full tiles")
    out = torch.empty((batch, n), device=x.device, dtype=x.dtype)
    partial = (torch.empty((batch, splits, n), device=x.device, dtype=torch.float32)
               if splits > 1 else out)
    _awq_gemv_kernel[(batch, triton.cdiv(n, block_n), splits)](
        x, qweight, qzeros, scales, partial,
        x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1), scales.stride(0), scales.stride(1),
        partial.stride(0), partial.stride(2) if splits > 1 else partial.stride(1),
        k, n, k // splits, block_k, block_n, num_warps=4,
    )
    if splits > 1:
        _merge_awq_splits[(batch, triton.cdiv(n, block_n))](
            partial, out, n, splits, block_n, num_warps=4,
        )
    return out
