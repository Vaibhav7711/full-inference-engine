"""Kernel-native FP8 (E4M3) paged KV primitives for the decode and prefill paths.

Same shape as :mod:`engine.kernels.int8_paged_kv`, one storage type apart: K/V vectors
are scaled per (token, head) and stored as `float8_e4m3fn`, and the attention kernels
multiply each vector back by its scale as it is loaded. Nothing materialises an FP16 copy
of the cache.

Why FP8 and not only INT8. On a card whose memory bandwidth is the decode floor - the
RTX 4060 measures 257 GB/s, the T4 258 - halving KV bytes per token (112 -> 56 KiB for
Qwen3-0.6B) is the one lever that moves the step. INT8 does the same arithmetic but has a
uniform grid; E4M3 spends three bits on exponent, so small values in a vector keep more
relative precision after a shared per-vector scale, which is the regime attention logits
live in. Whether that difference is measurable end to end is what the A/B is for; this
module only has to be correct and byte-equivalent in cost.

Hardware: the `float8e4nv` type exists in Triton for sm_89 and later. Ada has FP8 tensor
cores; here it is used only as a storage format, so the dequantising load is what gates
availability, not the matmul. `unavailable_reason()` says so for older devices.

Scale convention: `scale = max(|x|) / 448` with `448` the largest finite E4M3 value, and
the quantised value is clamped to +-448 before the cast. E4M3 has no infinity - an
overflowing cast produces NaN - so the clamp is correctness, not tidiness.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The contract (dtype, bound, availability, reference quantiser) is Triton-free in
# `fp8_format` so it can be tested without a GPU; re-exported here for callers. E4M3_MAX
# is never read as a module global inside a kernel (the static checks forbid that) - the
# kernels take it as a constexpr argument.
from engine.kernels.fp8_format import (  # noqa: F401
    E4M3_MAX, FP8_DTYPE, dequantize_fp8_reference, quantize_fp8_reference, unavailable_reason,
)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

@triton.jit
def _write_decode_fp8_kv_kernel(
    key_ptr, value_ptr, key_pool_ptr, value_pool_ptr, key_scale_ptr,
    value_scale_ptr, block_tables_ptr, seq_lens_ptr,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_btb,
    stride_kpb, stride_kps, stride_kph, stride_kpd,
    stride_vpb, stride_vps, stride_vph, stride_vpd,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, FP8_MAX: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    position = tl.load(seq_lens_ptr + batch)
    logical_block = position // BLOCK_SIZE
    block_offset = position % BLOCK_SIZE
    physical_block = tl.load(block_tables_ptr + batch * stride_btb + logical_block)

    key = tl.load(key_ptr + batch * stride_kb + head * stride_kh + dims * stride_kd).to(tl.float32)
    value = tl.load(value_ptr + batch * stride_vb + head * stride_vh + dims * stride_vd).to(tl.float32)
    key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / FP8_MAX, 1.0e-8)
    value_scale = tl.maximum(tl.max(tl.abs(value), axis=0) / FP8_MAX, 1.0e-8)

    key_dst = (key_pool_ptr + physical_block * stride_kpb + block_offset * stride_kps
               + head * stride_kph + dims * stride_kpd)
    value_dst = (value_pool_ptr + physical_block * stride_vpb + block_offset * stride_vps
                 + head * stride_vph + dims * stride_vpd)
    # The float -> fp8 cast rounds to nearest even. E4M3 has no infinity: a value past
    # 448 becomes NaN, so the clamp is what makes the cast total.
    key_quantized = tl.maximum(tl.minimum(key / key_scale, FP8_MAX), -FP8_MAX)
    value_quantized = tl.maximum(tl.minimum(value / value_scale, FP8_MAX), -FP8_MAX)
    tl.store(key_dst, key_quantized.to(key_pool_ptr.dtype.element_ty))
    tl.store(value_dst, value_quantized.to(value_pool_ptr.dtype.element_ty))
    tl.store(key_scale_ptr + physical_block * stride_ksb + block_offset * stride_kss + head * stride_ksh, key_scale)
    tl.store(value_scale_ptr + physical_block * stride_vsb + block_offset * stride_vss + head * stride_vsh, value_scale)


@triton.jit
def _write_prefill_fp8_kv_kernel(
    key_ptr, value_ptr, key_pool_ptr, value_pool_ptr, key_scale_ptr,
    value_scale_ptr, block_tables_ptr, start_positions_ptr, chunk_lens_ptr,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_btb, stride_btt,
    stride_kpb, stride_kps, stride_kph, stride_kpd,
    stride_vpb, stride_vps, stride_vph, stride_vpd,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, FP8_MAX: tl.constexpr,
):
    batch = tl.program_id(0)
    token = tl.program_id(1)
    head = tl.program_id(2)
    dims = tl.arange(0, HEAD_DIM)
    valid_token = token < tl.load(chunk_lens_ptr + batch)
    position = tl.load(start_positions_ptr + batch) + token
    logical_block = position // BLOCK_SIZE
    block_offset = position % BLOCK_SIZE
    physical_block = tl.load(
        block_tables_ptr + batch * stride_btb + logical_block * stride_btt,
        mask=valid_token, other=0,
    )
    valid = valid_token & (dims < HEAD_DIM)
    key = tl.load(key_ptr + batch * stride_kb + head * stride_kh + token * stride_kt + dims * stride_kd,
                  mask=valid, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + batch * stride_vb + head * stride_vh + token * stride_vt + dims * stride_vd,
                    mask=valid, other=0.0).to(tl.float32)
    key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / FP8_MAX, 1.0e-8)
    value_scale = tl.maximum(tl.max(tl.abs(value), axis=0) / FP8_MAX, 1.0e-8)
    key_quantized = tl.maximum(tl.minimum(key / key_scale, FP8_MAX), -FP8_MAX)
    value_quantized = tl.maximum(tl.minimum(value / value_scale, FP8_MAX), -FP8_MAX)
    key_dst = key_pool_ptr + physical_block * stride_kpb + block_offset * stride_kps + head * stride_kph + dims * stride_kpd
    value_dst = value_pool_ptr + physical_block * stride_vpb + block_offset * stride_vps + head * stride_vph + dims * stride_vpd
    tl.store(key_dst, key_quantized.to(key_pool_ptr.dtype.element_ty), mask=valid)
    tl.store(value_dst, value_quantized.to(value_pool_ptr.dtype.element_ty), mask=valid)
    tl.store(key_scale_ptr + physical_block * stride_ksb + block_offset * stride_kss + head * stride_ksh,
             key_scale, mask=valid_token)
    tl.store(value_scale_ptr + physical_block * stride_vsb + block_offset * stride_vss + head * stride_vsh,
             value_scale, mask=valid_token)


def _check_pools(key_pool, value_pool, key_scales, value_scales, heads, head_dim, what):
    if FP8_DTYPE is None or key_pool.dtype is not FP8_DTYPE or value_pool.dtype is not FP8_DTYPE:
        raise ValueError(f"{what}: K/V pools must use torch.float8_e4m3fn")
    if key_pool.shape != value_pool.shape or key_pool.ndim != 4 or key_pool.shape[2:] != (heads, head_dim):
        raise ValueError(f"{what}: K/V pool geometry does not match incoming tensors")
    if key_scales.shape != key_pool.shape[:-1] or value_scales.shape != key_pool.shape[:-1]:
        raise ValueError(f"{what}: scale pools must have shape [blocks, block_size, heads]")
    if head_dim > 128 or head_dim != triton.next_power_of_2(head_dim):
        raise ValueError(f"{what}: supports power-of-two head dimensions up to 128")


def write_decode_fp8_kv(
    key: torch.Tensor, value: torch.Tensor, key_pool: torch.Tensor,
    value_pool: torch.Tensor, key_scales: torch.Tensor, value_scales: torch.Tensor,
    block_tables: torch.Tensor, seq_lens: torch.Tensor,
) -> None:
    """Quantise one decode K/V vector per row directly into paged FP8 storage."""
    if key.ndim != 4 or value.shape != key.shape or key.shape[2] != 1:
        raise ValueError("key/value must have matching [N,H,1,D] shapes")
    batch, heads, _, head_dim = key.shape
    _check_pools(key_pool, value_pool, key_scales, value_scales, heads, head_dim, "FP8 decode write")
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=key.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=key.device)
    _write_decode_fp8_kv_kernel[(batch, heads)](
        key, value, key_pool, value_pool, key_scales, value_scales, block_tables, seq_lens,
        key.stride(0), key.stride(1), key.stride(3),
        value.stride(0), value.stride(1), value.stride(3), block_tables.stride(0),
        *key_pool.stride(), *value_pool.stride(), *key_scales.stride(), *value_scales.stride(),
        BLOCK_SIZE=key_pool.shape[1], HEAD_DIM=head_dim, FP8_MAX=E4M3_MAX, num_warps=4,
    )


def write_prefill_fp8_kv_batched(
    key: torch.Tensor, value: torch.Tensor, key_pool: torch.Tensor,
    value_pool: torch.Tensor, key_scales: torch.Tensor, value_scales: torch.Tensor,
    block_tables: torch.Tensor, chunk_lens: torch.Tensor,
    start_positions: torch.Tensor | None = None,
) -> None:
    """Quantise padded prefill chunks directly into FP8 paged K/V storage."""
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("key/value must have matching [N,H,T,D] shapes")
    batch, heads, padded_length, head_dim = key.shape
    if start_positions is None:
        start_positions = torch.zeros_like(chunk_lens)
    _check_pools(key_pool, value_pool, key_scales, value_scales, heads, head_dim, "FP8 prefill write")
    if block_tables.shape[0] != batch or chunk_lens.shape != (batch,) or start_positions.shape != (batch,):
        raise ValueError("invalid prefill metadata shapes")
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=key.device)
    chunk_lens = chunk_lens.contiguous().to(dtype=torch.int32, device=key.device)
    start_positions = start_positions.contiguous().to(dtype=torch.int32, device=key.device)
    _write_prefill_fp8_kv_kernel[(batch, padded_length, heads)](
        key, value, key_pool, value_pool, key_scales, value_scales,
        block_tables, start_positions, chunk_lens,
        *key.stride(), *value.stride(), *block_tables.stride(),
        *key_pool.stride(), *value_pool.stride(), *key_scales.stride(), *value_scales.stride(),
        BLOCK_SIZE=key_pool.shape[1], HEAD_DIM=head_dim, FP8_MAX=E4M3_MAX, num_warps=4,
    )


# ---------------------------------------------------------------------------
# Attention over FP8 pages
# ---------------------------------------------------------------------------

@triton.jit
def _paged_decode_batched_fp8_kernel(
    q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, out_ptr, bt_ptr, sl_ptr,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ksb, stride_kst, stride_ksh,
    stride_vsb, stride_vst, stride_vsh,
    stride_os, stride_oh, stride_od,
    stride_bts, stride_btt, stride_sl,
    num_q_heads, num_kv_heads, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    LENGTH_OFFSET: tl.constexpr,
):
    sequence = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (num_q_heads // num_kv_heads)
    dims = tl.arange(0, HEAD_DIM)
    offsets = tl.arange(0, BLOCK_N)
    sequence_length = tl.load(sl_ptr + sequence * stride_sl) + LENGTH_OFFSET
    query = tl.load(q_ptr + sequence * stride_qs + q_head * stride_qh + dims * stride_qd).to(tl.float32)
    table = bt_ptr + sequence * stride_bts

    max_value = float("-inf")
    normalizer = 0.0
    accumulator = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for start in range(0, sequence_length, BLOCK_N):
        positions = start + offsets
        valid = positions < sequence_length
        logical_blocks = positions // BLOCK_SIZE
        block_offsets = positions % BLOCK_SIZE
        physical_blocks = tl.load(table + logical_blocks * stride_btt, mask=valid, other=0)

        key_rows = (physical_blocks[:, None] * stride_kb + block_offsets[:, None] * stride_kt
                    + kv_head * stride_kh)
        key_scales = tl.load(
            ks_ptr + physical_blocks * stride_ksb + block_offsets * stride_kst + kv_head * stride_ksh,
            mask=valid, other=0.0,
        ).to(tl.float32)
        keys = tl.load(kp_ptr + key_rows + dims[None, :] * stride_kd,
                       mask=valid[:, None], other=0.0).to(tl.float32)
        keys = keys * key_scales[:, None]
        scores = tl.sum(query[None, :] * keys, axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_value, tile_max)
        alpha = tl.exp(max_value - new_max)
        probabilities = tl.exp(scores - new_max)

        value_rows = (physical_blocks[:, None] * stride_vb + block_offsets[:, None] * stride_vt
                      + kv_head * stride_vh)
        value_scales = tl.load(
            vs_ptr + physical_blocks * stride_vsb + block_offsets * stride_vst + kv_head * stride_vsh,
            mask=valid, other=0.0,
        ).to(tl.float32)
        values = tl.load(vp_ptr + value_rows + dims[None, :] * stride_vd,
                         mask=valid[:, None], other=0.0).to(tl.float32)
        values = values * value_scales[:, None]
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=0)
        max_value = new_max

    output = accumulator / tl.where(normalizer > 0.0, normalizer, 1.0)
    tl.store(out_ptr + sequence * stride_os + q_head * stride_oh + dims * stride_od,
             output.to(out_ptr.dtype.element_ty))


def paged_decode_batched_fp8(
    query: torch.Tensor, key_pages: torch.Tensor, value_pages: torch.Tensor,
    key_scales: torch.Tensor, value_scales: torch.Tensor, block_tables: torch.Tensor,
    seq_lens: torch.Tensor, *, scale: float | None = None, block_n: int = 128,
    num_warps: int = 4, length_offset: int = 0,
) -> torch.Tensor:
    """Decode attention over FP8 pages, dequantising each vector inside the kernel."""
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError("query must have shape [N,H,1,D]")
    batch, q_heads, _, head_dim = query.shape
    _, block_size, kv_heads, kv_dim = key_pages.shape
    if head_dim != kv_dim or q_heads % kv_heads:
        raise ValueError("unsupported attention head geometry")
    _check_pools(key_pages, value_pages, key_scales, value_scales, kv_heads, head_dim, "FP8 decode")
    if block_n not in {64, 128} or num_warps not in {4, 8}:
        raise ValueError("supported FP8 regimes are 64/128 tokens with 4/8 warps")
    if scale is None:
        scale = head_dim ** -0.5

    query = query.contiguous()
    key_pages, value_pages = key_pages.contiguous(), value_pages.contiguous()
    key_scales, value_scales = key_scales.contiguous(), value_scales.contiguous()
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=query.device)
    output = torch.empty_like(query)
    query_view, output_view = query.view(batch, q_heads, head_dim), output.view(batch, q_heads, head_dim)
    _paged_decode_batched_fp8_kernel[(batch, q_heads)](
        query_view, key_pages, value_pages, key_scales, value_scales, output_view,
        block_tables, seq_lens,
        *query_view.stride(), *key_pages.stride(), *value_pages.stride(),
        *key_scales.stride(), *value_scales.stride(), *output_view.stride(),
        *block_tables.stride(), seq_lens.stride(0), q_heads, kv_heads, scale,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCK_N=block_n,
        LENGTH_OFFSET=length_offset, num_warps=num_warps,
    )
    return output


@triton.jit
def _paged_prefill_fp8_kernel(
    q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, out_ptr, bt_ptr, starts_ptr, chunks_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ksb, stride_kst, stride_ksh,
    stride_vsb, stride_vst, stride_vsh,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_btb, stride_btt,
    num_q_heads, num_kv_heads, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    query_token = tl.program_id(2)
    dims = tl.arange(0, HEAD_DIM)
    chunk_length = tl.load(chunks_ptr + batch)
    query_valid = query_token < chunk_length
    start_position = tl.load(starts_ptr + batch)
    kv_length = tl.where(query_valid, start_position + query_token + 1, 0)
    kv_head = q_head // (num_q_heads // num_kv_heads)
    query = tl.load(
        q_ptr + batch * stride_qb + q_head * stride_qh + query_token * stride_qt + dims * stride_qd,
        mask=query_valid, other=0.0,
    ).to(tl.float32)
    table = bt_ptr + batch * stride_btb
    offsets = tl.arange(0, BLOCK_N)
    max_value = float("-inf")
    normalizer = 0.0
    accumulator = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for start in range(0, kv_length, BLOCK_N):
        positions = start + offsets
        valid = positions < kv_length
        logical_blocks = positions // BLOCK_SIZE
        block_offsets = positions % BLOCK_SIZE
        physical_blocks = tl.load(table + logical_blocks * stride_btt, mask=valid, other=0)
        key_rows = physical_blocks[:, None] * stride_kb + block_offsets[:, None] * stride_kt + kv_head * stride_kh
        key_scales = tl.load(
            ks_ptr + physical_blocks * stride_ksb + block_offsets * stride_kst + kv_head * stride_ksh,
            mask=valid, other=0.0,
        ).to(tl.float32)
        keys = tl.load(kp_ptr + key_rows + dims[None, :] * stride_kd,
                       mask=valid[:, None], other=0.0).to(tl.float32) * key_scales[:, None]
        scores = tl.sum(query[None, :] * keys, axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_value, tile_max)
        alpha = tl.exp(max_value - new_max)
        probabilities = tl.exp(scores - new_max)
        value_rows = physical_blocks[:, None] * stride_vb + block_offsets[:, None] * stride_vt + kv_head * stride_vh
        value_scales = tl.load(
            vs_ptr + physical_blocks * stride_vsb + block_offsets * stride_vst + kv_head * stride_vsh,
            mask=valid, other=0.0,
        ).to(tl.float32)
        values = tl.load(vp_ptr + value_rows + dims[None, :] * stride_vd,
                         mask=valid[:, None], other=0.0).to(tl.float32) * value_scales[:, None]
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=0)
        max_value = new_max
    result = tl.where(query_valid, accumulator / tl.where(normalizer > 0.0, normalizer, 1.0), 0.0)
    tl.store(out_ptr + batch * stride_ob + q_head * stride_oh + query_token * stride_ot + dims * stride_od,
             result.to(out_ptr.dtype.element_ty))


def paged_prefill_fp8(
    query: torch.Tensor, key_pages: torch.Tensor, value_pages: torch.Tensor,
    key_scales: torch.Tensor, value_scales: torch.Tensor, block_tables: torch.Tensor,
    start_positions: torch.Tensor, chunk_lens: torch.Tensor, *, scale: float | None = None,
    block_n: int = 64,
) -> torch.Tensor:
    """Causal paged prefill attention over fused-dequantised FP8 K/V pages."""
    if query.ndim != 4:
        raise ValueError("query must have shape [N,H,T,D]")
    batch, q_heads, query_length, head_dim = query.shape
    _, block_size, kv_heads, kv_dim = key_pages.shape
    if head_dim != kv_dim or q_heads % kv_heads:
        raise ValueError("unsupported attention head geometry")
    _check_pools(key_pages, value_pages, key_scales, value_scales, kv_heads, head_dim, "FP8 prefill")
    if block_tables.shape[0] != batch or start_positions.shape != (batch,) or chunk_lens.shape != (batch,):
        raise ValueError("invalid prefill metadata shapes")
    if block_n not in {64, 128}:
        raise ValueError("block_n must be 64 or 128")
    if scale is None:
        scale = head_dim ** -0.5
    query = query.contiguous()
    key_pages, value_pages = key_pages.contiguous(), value_pages.contiguous()
    key_scales, value_scales = key_scales.contiguous(), value_scales.contiguous()
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    start_positions = start_positions.contiguous().to(dtype=torch.int32, device=query.device)
    chunk_lens = chunk_lens.contiguous().to(dtype=torch.int32, device=query.device)
    output = torch.empty_like(query)
    _paged_prefill_fp8_kernel[(batch, q_heads, query_length)](
        query, key_pages, value_pages, key_scales, value_scales, output,
        block_tables, start_positions, chunk_lens,
        *query.stride(), *key_pages.stride(), *value_pages.stride(),
        *key_scales.stride(), *value_scales.stride(), *output.stride(), *block_tables.stride(),
        q_heads, kv_heads, scale, BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCK_N=block_n,
        num_warps=4,
    )
    return output
