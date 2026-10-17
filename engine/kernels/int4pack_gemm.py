"""Tensor-core W4A16 GEMM over PyTorch's tinygemm `int4pack` layout, for prefill rows.

`torch.ops.aten._weight_int4pack_mm` is fast for decode GEMV but slow for prefill: on the
RTX 4060 a 128-token Qwen3-4B-AWQ prefill/fused step costs ~158 ms against a 14 ms decode
step, ~8x the FP16 tensor-core floor. This kernel reads the *same* packed buffers (no
second weight copy), dequantizes each tile in registers and uses `tl.dot`.

Layout (`_convert_weight_to_int4pack(w, innerKTiles=8)`, shape `[N/8, K/128, 32, 4]`
int32), derived empirically with bit-coded probes and checked to be a translation-
invariant bijection per 8 x 128 (N x K) tile:

    lane   = 4 * (n % 8) + (k % 8) // 2
    word   = (k % 128) // 32
    nibble = (k % 32) // 8 + 4 * (k % 2)

Dequantization matches tinygemm: `(q - 8) * scale + zero`, with `[K/128, N, 2]`
(scale, zero) parameters per 128-wide K group.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int4pack_gemm_kernel(
    x_ptr, w_ptr, qp_ptr, out_ptr, M, N, K,
    stride_xm, stride_xk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * num_n
    first_m = (pid // group_size) * GROUP_M
    rows = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % rows
    pid_n = (pid % group_size) // rows

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_tiles = K // 128
    n_tile = offs_n // 8
    n_local = offs_n % 8
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        k = start_k + offs_k                                   # [BK]
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=offs_m[:, None] < M, other=0.0)
        k_tile = k // 128
        k_local = k % 128
        lane = 4 * n_local[None, :] + (k_local[:, None] % 8) // 2          # [BK, BN]
        word = k_local[:, None] // 32
        nibble = (k_local[:, None] % 32) // 8 + 4 * (k_local[:, None] % 2)
        address = ((n_tile[None, :] * k_tiles + k_tile[:, None]) * 32 + lane) * 4 + word
        packed = tl.load(w_ptr + address, mask=offs_n[None, :] < N, other=0)
        q = ((packed >> (4 * nibble)) & 15).to(tl.float32)
        group = start_k // 128
        qp = qp_ptr + (group * N + offs_n) * 2
        scale = tl.load(qp, mask=offs_n < N, other=0.0).to(tl.float32)
        zero = tl.load(qp + 1, mask=offs_n < N, other=0.0).to(tl.float32)
        w = ((q - 8.0) * scale[None, :] + zero[None, :]).to(tl.float16)
        acc += tl.dot(x, w)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def int4pack_gemm(
    inputs: torch.Tensor, packed: torch.Tensor, qparams: torch.Tensor, *,
    block_m: int = 64, block_n: int = 64, block_k: int = 64, num_warps: int = 4,
    num_stages: int = 3, group_m: int = 8,
) -> torch.Tensor:
    """`inputs [M, K]` (FP16) times the int4pack weight `[N, K]`, transposed -> `[M, N]` FP16."""
    if inputs.ndim != 2 or packed.ndim != 4 or packed.dtype is not torch.int32:
        raise ValueError("inputs must be [M, K] and packed an int4pack [N/8, K/128, 32, 4] tensor")
    M, K = inputs.shape
    N = packed.shape[0] * 8
    if packed.shape[1] * 128 != K or tuple(packed.shape[2:]) != (32, 4):
        raise ValueError("packed weight does not match the input width (innerKTiles must be 8)")
    if qparams.shape != (K // 128, N, 2):
        raise ValueError("qparams must be [K/128, N, 2]")
    if 128 % block_k or block_n % 8:
        raise ValueError("BLOCK_K must divide the 128-wide group and BLOCK_N be a multiple of 8")
    inputs = inputs.to(torch.float16)
    output = torch.empty((M, N), dtype=torch.float16, device=inputs.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _int4pack_gemm_kernel[grid](
        inputs, packed, qparams.contiguous(), output, M, N, K,
        inputs.stride(0), inputs.stride(1), output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


@triton.jit
def _int4pack_gemm_v2_kernel(
    x_ptr, w_ptr, qp_ptr, out_ptr, M, N, K,
    stride_xm, stride_xk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
):
    """One 128-wide K group per step; coalesced packed loads; permuted activations.

    For a fixed (n_tile, k_tile) the 128 int32 words are contiguous as
    [n(8)][kq(4)][word(4)], so row n's 16 words load as one vector and unpack in their
    natural order k' = (kq*4 + word)*8 + j. The activation tile is gathered at the
    matching logical columns k = 32*word + 8*(j%4) + 2*kq + j//4, so the dot product is
    unchanged. Dequantization factors out of the reduction:
    x . ((q-8)*s + z) = s * (x . (q-8)) + z * sum(x), with q-8 exact in FP16.
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * num_n
    first_m = (pid // group_size) * GROUP_M
    rows = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % rows
    pid_n = (pid % group_size) // rows

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_tiles = K // 128
    words = tl.arange(0, 16)                         # c = kq*4 + word
    nibbles = tl.arange(0, 8)
    kp = tl.arange(0, 128)                           # natural packed order k'
    kq = kp // 32
    word = (kp // 8) % 4
    j = kp % 8
    logical_k = 32 * word + 8 * (j % 4) + 2 * kq + j // 4
    row_base = ((offs_n // 8) * k_tiles) * 128 + (offs_n % 8) * 16       # [BN]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, k_tiles):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (kt * 128 + logical_k)[None, :] * stride_xk,
                    mask=offs_m[:, None] < M, other=0.0)                  # [BM, 128] permuted
        packed = tl.load(w_ptr + (row_base + kt * 128)[:, None] + words[None, :],
                         mask=offs_n[:, None] < N, other=0)              # [BN, 16]
        q = (packed[:, :, None] >> (4 * nibbles)[None, None, :]) & 15     # [BN, 16, 8]
        q = tl.reshape(q, (BLOCK_N, 128)).to(tl.float16) - 8.0
        partial = tl.dot(x, tl.trans(q))                                  # [BM, BN] fp32
        qp = qp_ptr + (kt * N + offs_n) * 2
        scale = tl.load(qp, mask=offs_n < N, other=0.0).to(tl.float32)
        zero = tl.load(qp + 1, mask=offs_n < N, other=0.0).to(tl.float32)
        row_sum = tl.sum(x.to(tl.float32), axis=1)
        acc += partial * scale[None, :] + row_sum[:, None] * zero[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def int4pack_gemm_v2(
    inputs: torch.Tensor, packed: torch.Tensor, qparams: torch.Tensor, *,
    block_m: int = 64, block_n: int = 64, num_warps: int = 4, num_stages: int = 2,
    group_m: int = 8,
) -> torch.Tensor:
    """Same contract as `int4pack_gemm`, with coalesced packed loads (see the kernel)."""
    M, K = inputs.shape
    N = packed.shape[0] * 8
    if packed.shape[1] * 128 != K or tuple(packed.shape[2:]) != (32, 4) or qparams.shape != (K // 128, N, 2):
        raise ValueError("int4pack geometry mismatch (innerKTiles must be 8)")
    if block_n % 8 or block_m < 16 or block_n < 16:
        raise ValueError("BLOCK_N must be a multiple of 8; tl.dot tiles need >= 16")
    inputs = inputs.to(torch.float16)
    output = torch.empty((M, N), dtype=torch.float16, device=inputs.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _int4pack_gemm_v2_kernel[grid](
        inputs, packed, qparams.contiguous(), output, M, N, K,
        inputs.stride(0), inputs.stride(1), output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


@triton.jit
def _int4pack_dequant_kernel(
    w_ptr, qp_ptr, out_ptr, N, K,
    BLOCK_N: tl.constexpr,
):
    """Dequantize one 128-wide K group for BLOCK_N rows into an FP16 `[N, K]` tensor."""
    pid_n = tl.program_id(0)
    kt = tl.program_id(1)
    k_tiles = K // 128
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    words = tl.arange(0, 16)
    nibbles = tl.arange(0, 8)
    kp = tl.arange(0, 128)
    kq = kp // 32
    word = (kp // 8) % 4
    j = kp % 8
    logical_k = 32 * word + 8 * (j % 4) + 2 * kq + j // 4
    row_base = ((offs_n // 8) * k_tiles + kt) * 128 + (offs_n % 8) * 16
    packed = tl.load(w_ptr + row_base[:, None] + words[None, :], mask=offs_n[:, None] < N, other=0)
    q = (packed[:, :, None] >> (4 * nibbles)[None, None, :]) & 15
    q = tl.reshape(q, (BLOCK_N, 128)).to(tl.float32)
    qp = qp_ptr + (kt * N + offs_n) * 2
    scale = tl.load(qp, mask=offs_n < N, other=0.0).to(tl.float32)
    zero = tl.load(qp + 1, mask=offs_n < N, other=0.0).to(tl.float32)
    w = (q - 8.0) * scale[:, None] + zero[:, None]
    tl.store(out_ptr + offs_n[:, None] * K + (kt * 128 + logical_k)[None, :], w.to(tl.float16),
             mask=offs_n[:, None] < N)


def int4pack_dequantize(packed: torch.Tensor, qparams: torch.Tensor, out: torch.Tensor | None = None,
                        block_n: int = 64) -> torch.Tensor:
    """Exact FP16 `[N, K]` weight from int4pack buffers (same math as tinygemm)."""
    N, K = packed.shape[0] * 8, packed.shape[1] * 128
    if out is None:
        out = torch.empty((N, K), dtype=torch.float16, device=packed.device)
    _int4pack_dequant_kernel[(triton.cdiv(N, block_n), K // 128)](
        packed, qparams.contiguous(), out, N, K, BLOCK_N=block_n, num_warps=4)
    return out


_SCRATCH: dict[torch.device, torch.Tensor] = {}


def _scratch_for(packed: torch.Tensor) -> torch.Tensor:
    """One FP16 scratch per device, grown to the largest `[N, K]` seen; reused by every
    projection because the dequant and its GEMM complete in stream order before the
    next projection runs."""
    N, K = packed.shape[0] * 8, packed.shape[1] * 128
    buffer = _SCRATCH.get(packed.device)
    if buffer is None or buffer.numel() < N * K:
        buffer = torch.empty(N * K, dtype=torch.float16, device=packed.device)
        _SCRATCH[packed.device] = buffer
    return buffer[:N * K].view(N, K)


def int4pack_gemm_dequant_cublas(inputs: torch.Tensor, packed: torch.Tensor, qparams: torch.Tensor,
                                 scratch: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize to FP16 scratch, then cuBLAS. Wins over the fused kernel at large M:
    the GEMM is compute-bound there and cuBLAS's tensor-core pipeline beats Triton's,
    while the extra ~3 bytes/weight of scratch traffic is amortized over the rows.
    RTX 4060, Qwen3-4B layer: 2.80 vs 3.36 ms at 256 rows, 14.2 vs 25.9 ms at 2048."""
    weight = int4pack_dequantize(packed, qparams, scratch if scratch is not None else _scratch_for(packed))
    return inputs.to(torch.float16) @ weight.T


@triton.jit
def _int4pack_gemm_v3_kernel(
    x_ptr, w_ptr, qp_ptr, out_ptr, M, N, K,
    stride_xm, stride_xk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
    BITCAST: tl.constexpr,
):
    """v2 with the nibble converted by bit tricks instead of int-to-float.

    `0x6400 | nibble` is the FP16 bit pattern of `1024 + nibble`; subtracting 1032 gives
    `nibble - 8` exactly with one FP16 subtract and no integer-to-float conversion.
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * num_n
    first_m = (pid // group_size) * GROUP_M
    rows = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % rows
    pid_n = (pid % group_size) // rows

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_tiles = K // 128
    words = tl.arange(0, 16)
    nibbles = tl.arange(0, 8)
    kp = tl.arange(0, 128)
    kq = kp // 32
    word = (kp // 8) % 4
    j = kp % 8
    logical_k = 32 * word + 8 * (j % 4) + 2 * kq + j // 4
    row_base = ((offs_n // 8) * k_tiles) * 128 + (offs_n % 8) * 16
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, k_tiles):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (kt * 128 + logical_k)[None, :] * stride_xk,
                    mask=offs_m[:, None] < M, other=0.0)
        packed = tl.load(w_ptr + (row_base + kt * 128)[:, None] + words[None, :],
                         mask=offs_n[:, None] < N, other=0)
        q = (packed[:, :, None] >> (4 * nibbles)[None, None, :]) & 15
        q = tl.reshape(q, (BLOCK_N, 128))
        if BITCAST:
            q = ((q | 0x6400).to(tl.int16).to(tl.float16, bitcast=True)) - 1032.0
        else:
            q = q.to(tl.float16) - 8.0
        partial = tl.dot(x, tl.trans(q))
        qp = qp_ptr + (kt * N + offs_n) * 2
        scale = tl.load(qp, mask=offs_n < N, other=0.0).to(tl.float32)
        zero = tl.load(qp + 1, mask=offs_n < N, other=0.0).to(tl.float32)
        row_sum = tl.sum(x.to(tl.float32), axis=1)
        acc += partial * scale[None, :] + row_sum[:, None] * zero[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def int4pack_gemm_v3(inputs, packed, qparams, *, block_m=32, block_n=128, num_warps=4,
                     num_stages=2, group_m=1, bitcast=True):
    M, K = inputs.shape
    N = packed.shape[0] * 8
    inputs = inputs.to(torch.float16)
    output = torch.empty((M, N), dtype=torch.float16, device=inputs.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _int4pack_gemm_v3_kernel[grid](
        inputs, packed, qparams.contiguous(), output, M, N, K,
        inputs.stride(0), inputs.stride(1), output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, GROUP_M=group_m, BITCAST=bitcast,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


# Offline-searched dispatch on RTX 4060 for the Qwen3-4B-AWQ projections
# (results/rtx4060/int4pack_prefill_search_4b_v3.json). Row counts under the first bound
# keep tinygemm, which wins at decode width (0.88x at 16 rows for the fused kernel).
FUSED_MIN_ROWS = 64
DEQUANT_MIN_ROWS = 192
FUSED_CONFIG = dict(block_m=32, block_n=128, num_warps=4, num_stages=2, group_m=1, bitcast=True)


def int4pack_gemm_dispatched(inputs: torch.Tensor, packed: torch.Tensor, qparams: torch.Tensor, *,
                             fused_min_rows: int = FUSED_MIN_ROWS,
                             dequant_min_rows: int | None = DEQUANT_MIN_ROWS) -> torch.Tensor:
    """tinygemm below `fused_min_rows`, the fused v3 kernel up to `dequant_min_rows`,
    dequant+cuBLAS from there (None disables that tier)."""
    rows = inputs.shape[0]
    if rows < fused_min_rows:
        return torch.ops.aten._weight_int4pack_mm(inputs.to(torch.bfloat16), packed, 128, qparams)
    if dequant_min_rows is not None and rows >= dequant_min_rows:
        return int4pack_gemm_dequant_cublas(inputs, packed, qparams)
    return int4pack_gemm_v3(inputs, packed, qparams, **FUSED_CONFIG)
