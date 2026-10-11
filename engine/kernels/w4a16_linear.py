"""W4A16 linear: INT4 weights packed two per byte, fp16 activations, group scales.

Companion to :mod:`w8a16_linear` with half the weight bytes again. The format and its
reference dequantiser live in :mod:`w4a16_format` (Triton-free); this module is the
kernel that consumes it.

Why the K tile equals the group: one group of `GROUP` inputs is `GROUP/2` packed bytes
whose low nibbles are inputs `[0, GROUP/2)` and high nibbles `[GROUP/2, GROUP)`. Loading
that byte tile once and splitting it gives two contiguous `[GROUP/2, BLOCK_N]` weight
tiles with no shuffle, each multiplied by the matching half of the activation tile, and
exactly one scale per (n, group) applies to both halves. Dequantisation is
`(nibble - 8) * scale`, done in fp16 right before `tl.dot`, so the tensor cores see fp16
and the memory system sees 4 bits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels.w4a16_format import GROUP_SIZE, OFFSET, quantize_weight_w4_grouped  # noqa: F401


@triton.jit
def _w4a16_linear_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,          # packed weight [N, K // 2], uint8
    stride_sn, stride_sg,          # scales [N, K // GROUP], fp16
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr, ZERO: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    HALF: tl.constexpr = GROUP // 2
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)
    m_mask = offs_m < M
    n_mask = offs_n < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for group in range(0, K // GROUP):
        k0 = group * GROUP
        # Activation halves: inputs [k0, k0+HALF) and [k0+HALF, k0+GROUP).
        x_lo = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_h[None, :]) * stride_xk,
            mask=m_mask[:, None], other=0.0,
        )
        x_hi = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + HALF + offs_h[None, :]) * stride_xk,
            mask=m_mask[:, None], other=0.0,
        )
        # One byte tile [HALF, BLOCK_N]: byte j of this group holds inputs j and j + HALF.
        byte_index = group * HALF + offs_h
        packed = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + byte_index[:, None] * stride_wk,
            mask=n_mask[None, :], other=ZERO,
        )
        scale = tl.load(scale_ptr + offs_n * stride_sn + group * stride_sg,
                        mask=n_mask, other=0.0).to(tl.float16)
        low = ((packed & 15).to(tl.int16) - ZERO).to(tl.float16) * scale[None, :]
        high = ((packed >> 4).to(tl.int16) - ZERO).to(tl.float16) * scale[None, :]
        accumulator += tl.dot(x_lo, low)
        accumulator += tl.dot(x_hi, high)
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             accumulator.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


def w4a16_linear(inputs: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                 bias: torch.Tensor | None = None, *, group_size: int = GROUP_SIZE,
                 block_m: int = 16, block_n: int = 64, num_warps: int = 4,
                 kernel: str = "auto") -> torch.Tensor:
    """`inputs [M, K] fp16` x dequantised `packed/scales` -> `[M, N]`, dequant in-kernel.

    `kernel`: "auto" takes the `tl.dot`-free GEMV for M <= 8 (decode) and the tiled
    kernel above it; "gemv" / "dot" force one, for the benchmark and the tests.
    """
    if inputs.ndim != 2 or packed.ndim != 2 or packed.dtype is not torch.uint8:
        raise ValueError("inputs must be 2D fp16 and packed weights 2D uint8")
    if kernel not in {"auto", "gemv", "dot"}:
        raise ValueError(f"kernel must be auto, gemv or dot; got {kernel!r}")
    if kernel == "gemv" or (kernel == "auto" and inputs.shape[0] <= GEMV_MAX_ROWS):
        return w4a16_gemv(inputs, packed, scales, bias, group_size=group_size)
    M, K = inputs.shape
    N, packed_k = packed.shape
    if group_size % 32 or group_size < 32:
        raise ValueError("group_size must be a multiple of 32 (each half feeds a tl.dot)")
    if K % group_size or packed_k != K // 2:
        raise ValueError(f"packed width {packed_k} does not match K={K} with group {group_size}")
    if scales.shape != (N, K // group_size):
        raise ValueError("scales must have shape [N, K // group_size]")
    if bias is not None and bias.shape != (N,):
        raise ValueError("bias must have shape [N]")
    if block_m < 16 or block_n < 16:
        raise ValueError("tl.dot requires tiles of at least 16")
    output = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _w4a16_linear_kernel[grid](
        inputs, packed, scales, bias if bias is not None else output, output,
        M, N, K,
        inputs.stride(0), inputs.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=bias is not None, BLOCK_M=block_m, BLOCK_N=block_n,
        GROUP=group_size, ZERO=OFFSET, num_warps=num_warps,
    )
    return output


@triton.jit
def _w4a16_gemv_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr, M_PAD: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr, ZERO: tl.constexpr,
):
    """Decode-width W4A16: `out[m, n] = sum_k x[m, k] * w[n, k]` for M <= M_PAD rows.

    No `tl.dot`: on sm_75 Triton lowers it to scalar FMA and the linear kernel runs
    ~20x off the weight sweep. Each program owns BLOCK_N output rows, dequantises one
    group's byte tile once, and applies it to every input row by a broadcast multiply
    and a `tl.sum` over k, so weight bytes cross HBM once and the arithmetic is the
    unavoidable M multiply-adds per weight.
    """
    pid_n = tl.program_id(0)
    HALF: tl.constexpr = GROUP // 2
    offs_m = tl.arange(0, M_PAD)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for group in range(0, K // GROUP):
        k0 = group * GROUP
        packed = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + (group * HALF + offs_h[None, :]) * stride_wk,
            mask=n_mask[:, None], other=ZERO,
        )                                                                   # [BLOCK_N, HALF]
        scale = tl.load(scale_ptr + offs_n * stride_sn + group * stride_sg, mask=n_mask, other=0.0)
        w_lo = ((packed & 0x0F).to(tl.int8) - ZERO).to(tl.float16) * scale[:, None]
        w_hi = ((packed >> 4).to(tl.int8) - ZERO).to(tl.float16) * scale[:, None]
        x_lo = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_h[None, :]) * stride_xk,
            mask=m_mask[:, None], other=0.0,
        )                                                                   # [M_PAD, HALF]
        x_hi = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + HALF + offs_h[None, :]) * stride_xk,
            mask=m_mask[:, None], other=0.0,
        )
        acc += tl.sum(x_lo[:, None, :].to(tl.float32) * w_lo[None, :, :].to(tl.float32), axis=2)
        acc += tl.sum(x_hi[:, None, :].to(tl.float32) * w_hi[None, :, :].to(tl.float32), axis=2)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


GEMV_MAX_ROWS = 8


def w4a16_gemv(inputs: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               bias: torch.Tensor | None = None, *, group_size: int = GROUP_SIZE,
               block_n: int = 32, num_warps: int = 4) -> torch.Tensor:
    """The decode-width path: `inputs [M <= 8, K]` x packed weights, no `tl.dot`."""
    M, K = inputs.shape
    N = packed.shape[0]
    if M > GEMV_MAX_ROWS:
        raise ValueError(f"w4a16_gemv serves M <= {GEMV_MAX_ROWS}; got {M}")
    if K % group_size or packed.shape != (N, K // 2) or scales.shape != (N, K // group_size):
        raise ValueError("packed/scales do not match inputs with this group size")
    m_pad = max(1, triton.next_power_of_2(M))
    output = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)
    _w4a16_gemv_kernel[(triton.cdiv(N, block_n),)](
        inputs, packed, scales, bias if bias is not None else output, output,
        M, N, K,
        inputs.stride(0), inputs.stride(1), packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1), output.stride(0), output.stride(1),
        HAS_BIAS=bias is not None, M_PAD=m_pad, BLOCK_N=block_n,
        GROUP=group_size, ZERO=OFFSET, num_warps=num_warps,
    )
    return output


@triton.jit
def _w4_dequant_kernel(
    w_ptr, scale_ptr, out_ptr,
    N, K,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
    stride_on, stride_ok,
    GROUP: tl.constexpr, ZERO: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """out[n, k] = (nibble(n, k) - ZERO) * scale[n, k // GROUP] over one group x BLOCK_N rows.

    Grid (groups, cdiv(N, BLOCK_N)): one program dequantises one group of GROUP inputs for
    BLOCK_N output rows - the byte tile is loaded once and split into its halves, like
    the linear kernel, and the two halves are written to contiguous columns.
    """
    group = tl.program_id(0)
    pid_n = tl.program_id(1)
    HALF: tl.constexpr = GROUP // 2
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF)
    n_mask = offs_n < N
    packed = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + (group * HALF + offs_h[None, :]) * stride_wk,
        mask=n_mask[:, None], other=ZERO,
    )
    scale = tl.load(scale_ptr + offs_n * stride_sn + group * stride_sg, mask=n_mask, other=0.0)
    low = ((packed & 0x0F).to(tl.int8) - ZERO).to(tl.float16) * scale[:, None]
    high = ((packed >> 4).to(tl.int8) - ZERO).to(tl.float16) * scale[:, None]
    base = out_ptr + offs_n[:, None] * stride_on + (group * GROUP + offs_h[None, :]) * stride_ok
    tl.store(base, low, mask=n_mask[:, None])
    tl.store(base + HALF * stride_ok, high, mask=n_mask[:, None])


def w4_dequantize_into(
    out: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, *,
    group_size: int = GROUP_SIZE, block_n: int = 64, num_warps: int = 4,
) -> torch.Tensor:
    """Dequantise `(packed, scales)` into `out` (fp16 `[N, K]`, caller-owned) in one launch.

    The dense path of `W4A16Linear` runs inside captured graphs at prefill widths; done
    with torch ops it was ~50 graph nodes per projection and a private-pool allocation
    per slab, which on the 8B made each fused graph's executable ~10x the size. One
    kernel into a persistent scratch is two nodes per projection and no allocation.
    Bit-identical to `unpack_w4_reference`: integer times fp16 scale, rounded once.
    """
    N, K = out.shape
    if out.dtype is not torch.float16 or packed.dtype is not torch.uint8:
        raise ValueError("out must be fp16 and packed uint8")
    if group_size % 2 or K % group_size or packed.shape != (N, K // 2):
        raise ValueError(f"packed {tuple(packed.shape)} does not match out {tuple(out.shape)} "
                         f"with group {group_size}")
    if scales.shape != (N, K // group_size):
        raise ValueError("scales must have shape [N, K // group_size]")
    grid = (K // group_size, triton.cdiv(N, block_n))
    _w4_dequant_kernel[grid](
        packed, scales, out, N, K,
        packed.stride(0), packed.stride(1), scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        GROUP=group_size, ZERO=OFFSET, BLOCK_N=block_n, num_warps=num_warps,
    )
    return out
