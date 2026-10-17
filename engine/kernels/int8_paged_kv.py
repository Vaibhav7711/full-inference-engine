"""Kernel-native INT8 paged KV primitives for the decode path.

Unlike the reference cache in :mod:`engine.quantization.kv_int8`, this module never
materializes a full FP16 copy of the cache.  K/V vectors are quantized while being
written, and the paged attention kernel multiplies each INT8 vector by its scale as it
is loaded.  It is intentionally an isolated kernel layer until its accuracy and
long-context latency gates are accepted.

Optional key smoothing: Qwen3 keys carry a few outlier channels (post-``k_norm`` and
RoPE) up to ~80x the median, so a per-token absmax scale spends most of the INT8 range
on them and the remaining channels lose precision.  With a static per-(kv head,
channel) factor ``s``, the write kernels store ``quantize(k / s)`` and the attention
kernels multiply the query by ``s`` instead: ``(q * s) . (k / s) == q . k``.  Storage
layout and per-token scales are unchanged.  ``s`` must be equal across each RoPE pair
(``i`` and ``i + D/2``); see :mod:`engine.quantization.kv_smoothing`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _write_decode_int8_kv_kernel(
    key_ptr, value_ptr, key_pool_ptr, value_pool_ptr, key_scale_ptr,
    value_scale_ptr, block_tables_ptr, seq_lens_ptr, smooth_ptr, stride_smh,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_btb,
    stride_kpb, stride_kps, stride_kph, stride_kpd,
    stride_vpb, stride_vps, stride_vph, stride_vpd,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, HAS_SMOOTH: tl.constexpr,
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
    if HAS_SMOOTH:
        key = key / tl.load(smooth_ptr + head * stride_smh + dims)
    key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1.0e-8)
    value_scale = tl.maximum(tl.max(tl.abs(value), axis=0) / 127.0, 1.0e-8)

    key_dst = (key_pool_ptr + physical_block * stride_kpb + block_offset * stride_kps
               + head * stride_kph + dims * stride_kpd)
    value_dst = (value_pool_ptr + physical_block * stride_vpb + block_offset * stride_vps
                 + head * stride_vph + dims * stride_vpd)
    # Triton's float-to-int cast truncates.  Make round-to-nearest explicit so this
    # kernel has the same quantization contract as torch.round in the reference path.
    key_quantized = key / key_scale
    value_quantized = value / value_scale
    key_quantized = tl.where(key_quantized >= 0.0, key_quantized + 0.5, key_quantized - 0.5)
    value_quantized = tl.where(value_quantized >= 0.0, value_quantized + 0.5, value_quantized - 0.5)
    key_quantized = tl.maximum(tl.minimum(key_quantized, 127.0), -127.0)
    value_quantized = tl.maximum(tl.minimum(value_quantized, 127.0), -127.0)
    tl.store(key_dst, key_quantized.to(tl.int8))
    tl.store(value_dst, value_quantized.to(tl.int8))
    tl.store(key_scale_ptr + physical_block * stride_ksb + block_offset * stride_kss + head * stride_ksh, key_scale)
    tl.store(value_scale_ptr + physical_block * stride_vsb + block_offset * stride_vss + head * stride_vsh, value_scale)


@triton.jit
def _paged_decode_batched_int8_kernel(
    q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, out_ptr, bt_ptr, sl_ptr, smooth_ptr, stride_smh,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ksb, stride_kst, stride_ksh,
    stride_vsb, stride_vst, stride_vsh,
    stride_os, stride_oh, stride_od,
    stride_bts, stride_btt, stride_sl,
    num_q_heads, num_kv_heads, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    LENGTH_OFFSET: tl.constexpr, HAS_SMOOTH: tl.constexpr,
):
    sequence = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (num_q_heads // num_kv_heads)
    dims = tl.arange(0, HEAD_DIM)
    offsets = tl.arange(0, BLOCK_N)
    sequence_length = tl.load(sl_ptr + sequence * stride_sl) + LENGTH_OFFSET
    query = tl.load(q_ptr + sequence * stride_qs + q_head * stride_qh + dims * stride_qd).to(tl.float32)
    if HAS_SMOOTH:
        query = query * tl.load(smooth_ptr + kv_head * stride_smh + dims)
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
                       mask=valid[:, None], other=0).to(tl.float32)
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
                         mask=valid[:, None], other=0).to(tl.float32)
        values = values * value_scales[:, None]
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=0)
        max_value = new_max

    output = accumulator / tl.where(normalizer > 0.0, normalizer, 1.0)
    tl.store(out_ptr + sequence * stride_os + q_head * stride_oh + dims * stride_od,
             output.to(out_ptr.dtype.element_ty))


def _smoothing_arg(key_smoothing: torch.Tensor | None, heads: int, head_dim: int,
                   device, fallback: torch.Tensor) -> tuple[torch.Tensor, int, bool]:
    """Validate an optional `[kv_heads, head_dim]` FP32 key-smoothing factor."""
    if key_smoothing is None:
        return fallback, 0, False
    if (key_smoothing.shape != (heads, head_dim) or key_smoothing.dtype is not torch.float32
            or key_smoothing.device != torch.device(device) or not key_smoothing.is_contiguous()):
        raise ValueError("key_smoothing must be a contiguous FP32 [kv_heads, head_dim] "
                         "tensor on the cache device")
    return key_smoothing, key_smoothing.stride(0), True


def write_decode_int8_kv(
    key: torch.Tensor, value: torch.Tensor, key_pool: torch.Tensor,
    value_pool: torch.Tensor, key_scales: torch.Tensor, value_scales: torch.Tensor,
    block_tables: torch.Tensor, seq_lens: torch.Tensor,
    key_smoothing: torch.Tensor | None = None,
) -> None:
    """Quantize one decode K/V vector per row directly into paged INT8 storage."""
    if key.ndim != 4 or value.shape != key.shape or key.shape[2] != 1:
        raise ValueError("key/value must have matching [N,H,1,D] shapes")
    batch, heads, _, head_dim = key.shape
    if head_dim > 128 or head_dim != triton.next_power_of_2(head_dim):
        raise ValueError("INT8 paged decode supports power-of-two head dimensions up to 128")
    if key_pool.dtype is not torch.int8 or value_pool.dtype is not torch.int8:
        raise ValueError("INT8 K/V pools must use torch.int8")
    if key_pool.shape != value_pool.shape or key_pool.shape[2:] != (heads, head_dim):
        raise ValueError("INT8 K/V pool geometry does not match incoming tensors")
    if key_scales.shape != key_pool.shape[:-1] or value_scales.shape != key_pool.shape[:-1]:
        raise ValueError("scale pools must have shape [blocks, block_size, heads]")

    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=key.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=key.device)
    smooth, stride_smh, has_smooth = _smoothing_arg(key_smoothing, heads, head_dim, key.device, key_scales)
    _write_decode_int8_kv_kernel[(batch, heads)](
        key, value, key_pool, value_pool, key_scales, value_scales, block_tables, seq_lens,
        smooth, stride_smh, key.stride(0), key.stride(1), key.stride(3),
        value.stride(0), value.stride(1), value.stride(3), block_tables.stride(0),
        *key_pool.stride(), *value_pool.stride(), *key_scales.stride(), *value_scales.stride(),
        BLOCK_SIZE=key_pool.shape[1], HEAD_DIM=head_dim, HAS_SMOOTH=has_smooth, num_warps=4,
    )


@triton.jit
def _write_prefill_int8_kv_kernel(
    key_ptr, value_ptr, key_pool_ptr, value_pool_ptr, key_scale_ptr,
    value_scale_ptr, block_tables_ptr, start_positions_ptr, chunk_lens_ptr,
    smooth_ptr, stride_smh,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_btb, stride_btt,
    stride_kpb, stride_kps, stride_kph, stride_kpd,
    stride_vpb, stride_vps, stride_vph, stride_vpd,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, HAS_SMOOTH: tl.constexpr,
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
    if HAS_SMOOTH:
        key = key / tl.load(smooth_ptr + head * stride_smh + dims)
    key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1.0e-8)
    value_scale = tl.maximum(tl.max(tl.abs(value), axis=0) / 127.0, 1.0e-8)
    key_quantized = key / key_scale
    value_quantized = value / value_scale
    key_quantized = tl.where(key_quantized >= 0.0, key_quantized + 0.5, key_quantized - 0.5)
    value_quantized = tl.where(value_quantized >= 0.0, value_quantized + 0.5, value_quantized - 0.5)
    key_quantized = tl.maximum(tl.minimum(key_quantized, 127.0), -127.0)
    value_quantized = tl.maximum(tl.minimum(value_quantized, 127.0), -127.0)
    key_dst = key_pool_ptr + physical_block * stride_kpb + block_offset * stride_kps + head * stride_kph + dims * stride_kpd
    value_dst = value_pool_ptr + physical_block * stride_vpb + block_offset * stride_vps + head * stride_vph + dims * stride_vpd
    tl.store(key_dst, key_quantized.to(tl.int8), mask=valid)
    tl.store(value_dst, value_quantized.to(tl.int8), mask=valid)
    tl.store(key_scale_ptr + physical_block * stride_ksb + block_offset * stride_kss + head * stride_ksh,
             key_scale, mask=valid_token)
    tl.store(value_scale_ptr + physical_block * stride_vsb + block_offset * stride_vss + head * stride_vsh,
             value_scale, mask=valid_token)


def write_prefill_int8_kv_batched(
    key: torch.Tensor, value: torch.Tensor, key_pool: torch.Tensor,
    value_pool: torch.Tensor, key_scales: torch.Tensor, value_scales: torch.Tensor,
    block_tables: torch.Tensor, chunk_lens: torch.Tensor,
    start_positions: torch.Tensor | None = None,
    key_smoothing: torch.Tensor | None = None,
) -> None:
    """Quantize padded prefill chunks directly into INT8 paged K/V storage."""
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("key/value must have matching [N,H,T,D] shapes")
    batch, heads, padded_length, head_dim = key.shape
    if start_positions is None:
        start_positions = torch.zeros_like(chunk_lens)
    if head_dim > 128 or head_dim != triton.next_power_of_2(head_dim):
        raise ValueError("INT8 paged prefill supports power-of-two head dimensions up to 128")
    if (key_pool.dtype is not torch.int8 or value_pool.dtype is not torch.int8
            or key_pool.shape != value_pool.shape or key_pool.shape[2:] != (heads, head_dim)):
        raise ValueError("INT8 K/V pool geometry does not match incoming tensors")
    if key_scales.shape != key_pool.shape[:-1] or value_scales.shape != key_pool.shape[:-1]:
        raise ValueError("scale pools must have shape [blocks, block_size, heads]")
    if block_tables.shape[0] != batch or chunk_lens.shape != (batch,) or start_positions.shape != (batch,):
        raise ValueError("invalid prefill metadata shapes")

    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=key.device)
    chunk_lens = chunk_lens.contiguous().to(dtype=torch.int32, device=key.device)
    start_positions = start_positions.contiguous().to(dtype=torch.int32, device=key.device)
    smooth, stride_smh, has_smooth = _smoothing_arg(key_smoothing, heads, head_dim, key.device, key_scales)
    _write_prefill_int8_kv_kernel[(batch, padded_length, heads)](
        key, value, key_pool, value_pool, key_scales, value_scales,
        block_tables, start_positions, chunk_lens, smooth, stride_smh,
        *key.stride(), *value.stride(), *block_tables.stride(),
        *key_pool.stride(), *value_pool.stride(), *key_scales.stride(), *value_scales.stride(),
        BLOCK_SIZE=key_pool.shape[1], HEAD_DIM=head_dim, HAS_SMOOTH=has_smooth, num_warps=4,
    )


def paged_decode_batched_int8(
    query: torch.Tensor, key_pages: torch.Tensor, value_pages: torch.Tensor,
    key_scales: torch.Tensor, value_scales: torch.Tensor, block_tables: torch.Tensor,
    seq_lens: torch.Tensor, *, scale: float | None = None, block_n: int = 128,
    num_warps: int = 4, length_offset: int = 0, key_smoothing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention over INT8 pages, dequantizing each vector inside the kernel."""
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError("query must have shape [N,H,1,D]")
    batch, q_heads, _, head_dim = query.shape
    if key_pages.dtype is not torch.int8 or value_pages.dtype is not torch.int8:
        raise ValueError("INT8 K/V pools must use torch.int8")
    if key_pages.shape != value_pages.shape or key_pages.ndim != 4:
        raise ValueError("K/V pools must have matching [blocks,block,H,D] shapes")
    _, block_size, kv_heads, kv_dim = key_pages.shape
    if head_dim != kv_dim or head_dim > 128 or q_heads % kv_heads:
        raise ValueError("unsupported attention head geometry")
    if key_scales.shape != key_pages.shape[:-1] or value_scales.shape != key_pages.shape[:-1]:
        raise ValueError("scale pools must have shape [blocks, block_size, kv_heads]")
    if block_n not in {64, 128} or num_warps not in {4, 8}:
        raise ValueError("supported INT8 regimes are 64/128 tokens with 4/8 warps")
    if scale is None:
        scale = head_dim ** -0.5

    query = query.contiguous()
    key_pages, value_pages = key_pages.contiguous(), value_pages.contiguous()
    key_scales, value_scales = key_scales.contiguous(), value_scales.contiguous()
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=query.device)
    output = torch.empty_like(query)
    query_view, output_view = query.view(batch, q_heads, head_dim), output.view(batch, q_heads, head_dim)
    smooth, stride_smh, has_smooth = _smoothing_arg(key_smoothing, kv_heads, head_dim, query.device, key_scales)
    _paged_decode_batched_int8_kernel[(batch, q_heads)](
        query_view, key_pages, value_pages, key_scales, value_scales, output_view,
        block_tables, seq_lens, smooth, stride_smh,
        *query_view.stride(), *key_pages.stride(), *value_pages.stride(),
        *key_scales.stride(), *value_scales.stride(), *output_view.stride(),
        *block_tables.stride(), seq_lens.stride(0), q_heads, kv_heads, scale,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCK_N=block_n,
        LENGTH_OFFSET=length_offset, HAS_SMOOTH=has_smooth, num_warps=num_warps,
    )
    return output


@triton.jit
def _paged_prefill_int8_kernel(
    q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, out_ptr, bt_ptr, starts_ptr, chunks_ptr,
    smooth_ptr, stride_smh,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ksb, stride_kst, stride_ksh,
    stride_vsb, stride_vst, stride_vsh,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_btb, stride_btt,
    num_q_heads, num_kv_heads, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_SMOOTH: tl.constexpr,
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
    if HAS_SMOOTH:
        query = query * tl.load(smooth_ptr + kv_head * stride_smh + dims)
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
                       mask=valid[:, None], other=0).to(tl.float32) * key_scales[:, None]
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
                         mask=valid[:, None], other=0).to(tl.float32) * value_scales[:, None]
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=0)
        max_value = new_max
    result = tl.where(query_valid, accumulator / tl.where(normalizer > 0.0, normalizer, 1.0), 0.0)
    tl.store(out_ptr + batch * stride_ob + q_head * stride_oh + query_token * stride_ot + dims * stride_od,
             result.to(out_ptr.dtype.element_ty))


def paged_prefill_int8(
    query: torch.Tensor, key_pages: torch.Tensor, value_pages: torch.Tensor,
    key_scales: torch.Tensor, value_scales: torch.Tensor, block_tables: torch.Tensor,
    start_positions: torch.Tensor, chunk_lens: torch.Tensor, *, scale: float | None = None,
    block_n: int = 64, key_smoothing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal paged prefill attention over fused-dequantized INT8 K/V pages."""
    if query.ndim != 4:
        raise ValueError("query must have shape [N,H,T,D]")
    batch, q_heads, query_length, head_dim = query.shape
    if (key_pages.dtype is not torch.int8 or value_pages.dtype is not torch.int8
            or key_pages.shape != value_pages.shape or key_pages.ndim != 4):
        raise ValueError("K/V pools must be matching INT8 [blocks,block,H,D] tensors")
    _, block_size, kv_heads, kv_dim = key_pages.shape
    if head_dim != kv_dim or head_dim > 128 or head_dim != triton.next_power_of_2(head_dim) or q_heads % kv_heads:
        raise ValueError("unsupported attention head geometry")
    if key_scales.shape != key_pages.shape[:-1] or value_scales.shape != key_pages.shape[:-1]:
        raise ValueError("scale pools must have shape [blocks, block_size, heads]")
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
    smooth, stride_smh, has_smooth = _smoothing_arg(key_smoothing, kv_heads, head_dim, query.device, key_scales)
    _paged_prefill_int8_kernel[(batch, q_heads, query_length)](
        query, key_pages, value_pages, key_scales, value_scales, output,
        block_tables, start_positions, chunk_lens, smooth, stride_smh,
        *query.stride(), *key_pages.stride(), *value_pages.stride(),
        *key_scales.stride(), *value_scales.stride(), *output.stride(), *block_tables.stride(),
        q_heads, kv_heads, scale, BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCK_N=block_n,
        HAS_SMOOTH=has_smooth, num_warps=4,
    )
    return output


@triton.jit
def _tiled_paged_prefill_int8_kernel(
    q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, out_ptr, bt_ptr, starts_ptr, chunks_ptr,
    smooth_ptr, stride_smh,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ksb, stride_kst, stride_ksh,
    stride_vsb, stride_vst, stride_vsh,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_btb, stride_btt,
    num_q_heads, num_kv_heads, scale, max_blocks, query_len,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, HAS_SMOOTH: tl.constexpr,
):
    """FlashAttention-2 tile over INT8 pages: one program per (row, head, query tile).

    INT8 values convert exactly to FP16, so both products run on tensor cores. The key
    scale is per (token, head): applied to score columns after `q . k_int`, which costs a
    [BLOCK_M, BLOCK_N] multiply instead of dequantizing the [BLOCK_N, HEAD_DIM] tile.
    """
    neg = -1.0e30
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    tile = tl.program_id(2)
    chunk_len = tl.load(chunks_ptr + batch)
    start = tl.load(starts_ptr + batch)
    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    row_valid = offs_m < chunk_len
    in_tensor = offs_m < query_len
    q_pos = start + offs_m
    kv_head = q_head // (num_q_heads // num_kv_heads)

    q = tl.load(q_ptr + batch * stride_qb + q_head * stride_qh
                + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=row_valid[:, None], other=0.0).to(tl.float32)
    if HAS_SMOOTH:
        q = q * tl.load(smooth_ptr + kv_head * stride_smh + offs_d)[None, :]
    q = (q * scale).to(tl.float16)

    m_i = tl.full([BLOCK_M], neg, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    tile_last = tl.minimum(tile * BLOCK_M + BLOCK_M, chunk_len)
    # Padding-only tiles visit no keys (see tiled_paged_prefill).
    kv_end = tl.where(tile * BLOCK_M < chunk_len, start + tile_last, 0)
    table = bt_ptr + batch * stride_btb
    for start_n in range(0, kv_end, BLOCK_N):
        positions = start_n + offs_n
        in_range = (positions < kv_end) & (positions // BLOCK_SIZE < max_blocks)
        logical = positions // BLOCK_SIZE
        offsets = positions % BLOCK_SIZE
        physical = tl.load(table + logical * stride_btt, mask=in_range, other=0)
        valid_n = in_range & (physical >= 0)
        rows_k = physical * stride_kb + offsets * stride_kt + kv_head * stride_kh
        keys = tl.load(kp_ptr + rows_k[:, None] + offs_d[None, :] * stride_kd,
                       mask=valid_n[:, None], other=0).to(tl.float16)
        key_scales = tl.load(ks_ptr + physical * stride_ksb + offsets * stride_kst + kv_head * stride_ksh,
                             mask=valid_n, other=0.0).to(tl.float32)
        scores = tl.dot(q, tl.trans(keys)) * key_scales[None, :]
        visible = (positions[None, :] <= q_pos[:, None]) & valid_n[None, :] & row_valid[:, None]
        scores = tl.where(visible, scores, neg)
        new_max = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - new_max)
        probs = tl.where(visible, tl.exp(scores - new_max[:, None]), 0.0)
        rows_v = physical * stride_vb + offsets * stride_vt + kv_head * stride_vh
        value_scales = tl.load(vs_ptr + physical * stride_vsb + offsets * stride_vst + kv_head * stride_vsh,
                               mask=valid_n, other=0.0).to(tl.float32)
        values = tl.load(vp_ptr + rows_v[:, None] + offs_d[None, :] * stride_vd,
                         mask=valid_n[:, None], other=0).to(tl.float32)
        values = (values * value_scales[:, None]).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.float16), values)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = new_max
    result = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    result = tl.where(row_valid[:, None], result, 0.0)
    tl.store(out_ptr + batch * stride_ob + q_head * stride_oh
             + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
             result.to(out_ptr.dtype.element_ty), mask=in_tensor[:, None])


def select_int8_prefill_config(query_len: int) -> tuple[int, int, int, int]:
    """Offline-searched `(BLOCK_M, BLOCK_N, num_warps, num_stages)` for RTX 4060.

    From `results/rtx4060/int8_prefill_search.json` (rows 1/2/4, chunks 16-128, contexts
    0-4096, Qwen3 16/8 heads): 16x128 is within 7% of the per-cell best for chunks up to
    32; 32x128 within 8% for 64/128-token chunks at contexts of 1k and more. Selected
    from the static padded width, so a captured graph never reselects or compiles.
    """
    if query_len <= 32:
        return 16, 128, 4, 2
    return 32, 128, 4, 2


def tiled_paged_prefill_int8(
    query: torch.Tensor, key_pages: torch.Tensor, value_pages: torch.Tensor,
    key_scales: torch.Tensor, value_scales: torch.Tensor, block_tables: torch.Tensor,
    start_positions: torch.Tensor, chunk_lens: torch.Tensor, *, scale: float | None = None,
    key_smoothing: torch.Tensor | None = None, block_m: int | None = None,
    block_n: int | None = None, num_warps: int | None = None, num_stages: int | None = None,
) -> torch.Tensor:
    """Tensor-core causal paged prefill over INT8 K/V; same contract as `paged_prefill_int8`."""
    if query.ndim != 4:
        raise ValueError("query must have shape [N,H,T,D]")
    batch, q_heads, query_length, head_dim = query.shape
    if (key_pages.dtype is not torch.int8 or value_pages.dtype is not torch.int8
            or key_pages.shape != value_pages.shape or key_pages.ndim != 4):
        raise ValueError("K/V pools must be matching INT8 [blocks,block,H,D] tensors")
    _, block_size, kv_heads, kv_dim = key_pages.shape
    if head_dim != kv_dim or head_dim > 128 or head_dim != triton.next_power_of_2(head_dim) or q_heads % kv_heads:
        raise ValueError("unsupported attention head geometry")
    if key_scales.shape != key_pages.shape[:-1] or value_scales.shape != key_pages.shape[:-1]:
        raise ValueError("scale pools must have shape [blocks, block_size, heads]")
    if block_tables.shape[0] != batch or start_positions.shape != (batch,) or chunk_lens.shape != (batch,):
        raise ValueError("invalid prefill metadata shapes")
    default_m, default_n, default_warps, default_stages = select_int8_prefill_config(query_length)
    block_m = default_m if block_m is None else block_m
    block_n = default_n if block_n is None else block_n
    num_warps = default_warps if num_warps is None else num_warps
    num_stages = default_stages if num_stages is None else num_stages
    if block_m < 16 or block_n < 16:
        raise ValueError("tl.dot requires tiles of at least 16")
    if scale is None:
        scale = head_dim ** -0.5
    query = query.contiguous()
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    start_positions = start_positions.contiguous().to(dtype=torch.int32, device=query.device)
    chunk_lens = chunk_lens.contiguous().to(dtype=torch.int32, device=query.device)
    output = torch.empty_like(query)
    smooth, stride_smh, has_smooth = _smoothing_arg(key_smoothing, kv_heads, head_dim, query.device, key_scales)
    grid = (batch, q_heads, triton.cdiv(query_length, block_m))
    _tiled_paged_prefill_int8_kernel[grid](
        query, key_pages, value_pages, key_scales, value_scales, output,
        block_tables, start_positions, chunk_lens, smooth, stride_smh,
        *query.stride(), *key_pages.stride(), *value_pages.stride(),
        *key_scales.stride(), *value_scales.stride(), *output.stride(), *block_tables.stride(),
        q_heads, kv_heads, scale, block_tables.shape[1], query_length,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        HAS_SMOOTH=has_smooth, num_warps=num_warps, num_stages=num_stages,
    )
    return output
