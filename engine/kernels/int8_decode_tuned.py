"""Search-space INT8 paged decode attention: pipelining, tile width and paired GQA.

`paged_decode_batched_int8` runs one program per (row, query head), so with Qwen3's
two query heads per KV head every INT8 K/V tile and its scales are loaded twice. The
paired variant runs one program per (row, KV head) and scores both query heads against
one load. Same math and accumulation order per head as the per-head kernel; this module
is an offline-search candidate until an end-to-end A/B accepts a dispatch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_decode_pair_kernel(
    q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, out_ptr, bt_ptr, sl_ptr, smooth_ptr, stride_smh,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ksb, stride_kst, stride_ksh,
    stride_vsb, stride_vst, stride_vsh,
    stride_os, stride_oh, stride_od,
    stride_bts, stride_btt, stride_sl, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    LENGTH_OFFSET: tl.constexpr, HAS_SMOOTH: tl.constexpr,
):
    sequence = tl.program_id(0)
    kv_head = tl.program_id(1)
    head0 = 2 * kv_head
    dims = tl.arange(0, HEAD_DIM)
    offsets = tl.arange(0, BLOCK_N)
    sequence_length = tl.load(sl_ptr + sequence * stride_sl) + LENGTH_OFFSET
    q_base = q_ptr + sequence * stride_qs + dims * stride_qd
    q0 = tl.load(q_base + head0 * stride_qh).to(tl.float32)
    q1 = tl.load(q_base + (head0 + 1) * stride_qh).to(tl.float32)
    if HAS_SMOOTH:
        smooth = tl.load(smooth_ptr + kv_head * stride_smh + dims)
        q0 = q0 * smooth
        q1 = q1 * smooth
    table = bt_ptr + sequence * stride_bts
    m0 = float("-inf")
    m1 = float("-inf")
    l0 = 0.0
    l1 = 0.0
    a0 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    a1 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for start in range(0, sequence_length, BLOCK_N):
        positions = start + offsets
        valid = positions < sequence_length
        physical = tl.load(table + (positions // BLOCK_SIZE) * stride_btt, mask=valid, other=0)
        slot = positions % BLOCK_SIZE
        k_scale = tl.load(ks_ptr + physical * stride_ksb + slot * stride_kst + kv_head * stride_ksh,
                          mask=valid, other=0.0).to(tl.float32)
        keys = tl.load(kp_ptr + physical[:, None] * stride_kb + slot[:, None] * stride_kt
                       + kv_head * stride_kh + dims[None, :] * stride_kd,
                       mask=valid[:, None], other=0).to(tl.float32) * k_scale[:, None]
        s0 = tl.where(valid, tl.sum(q0[None, :] * keys, axis=1) * scale, float("-inf"))
        s1 = tl.where(valid, tl.sum(q1[None, :] * keys, axis=1) * scale, float("-inf"))
        n0 = tl.maximum(m0, tl.max(s0, axis=0))
        n1 = tl.maximum(m1, tl.max(s1, axis=0))
        alpha0 = tl.exp(m0 - n0)
        alpha1 = tl.exp(m1 - n1)
        p0 = tl.exp(s0 - n0)
        p1 = tl.exp(s1 - n1)
        v_scale = tl.load(vs_ptr + physical * stride_vsb + slot * stride_vst + kv_head * stride_vsh,
                          mask=valid, other=0.0).to(tl.float32)
        values = tl.load(vp_ptr + physical[:, None] * stride_vb + slot[:, None] * stride_vt
                         + kv_head * stride_vh + dims[None, :] * stride_vd,
                         mask=valid[:, None], other=0).to(tl.float32) * v_scale[:, None]
        a0 = a0 * alpha0 + tl.sum(p0[:, None] * values, axis=0)
        a1 = a1 * alpha1 + tl.sum(p1[:, None] * values, axis=0)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        l1 = l1 * alpha1 + tl.sum(p1, axis=0)
        m0 = n0
        m1 = n1
    out_base = out_ptr + sequence * stride_os + dims * stride_od
    tl.store(out_base + head0 * stride_oh,
             (a0 / tl.where(l0 > 0.0, l0, 1.0)).to(out_ptr.dtype.element_ty))
    tl.store(out_base + (head0 + 1) * stride_oh,
             (a1 / tl.where(l1 > 0.0, l1, 1.0)).to(out_ptr.dtype.element_ty))


def paged_decode_int8_paired(
    query: torch.Tensor, key_pages: torch.Tensor, value_pages: torch.Tensor,
    key_scales: torch.Tensor, value_scales: torch.Tensor, block_tables: torch.Tensor,
    seq_lens: torch.Tensor, *, scale: float | None = None, block_n: int = 64,
    num_warps: int = 4, num_stages: int = 2, length_offset: int = 0,
    key_smoothing: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8 decode with one program per (row, KV head) for exactly two query heads per group."""
    from engine.kernels.int8_paged_kv import _smoothing_arg

    batch, q_heads, _, head_dim = query.shape
    _, block_size, kv_heads, _ = key_pages.shape
    if q_heads != 2 * kv_heads:
        raise ValueError("the paired INT8 decode kernel needs exactly two query heads per KV head")
    if scale is None:
        scale = head_dim ** -0.5
    query = query.contiguous()
    output = torch.empty_like(query)
    q_view, o_view = query.view(batch, q_heads, head_dim), output.view(batch, q_heads, head_dim)
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=query.device)
    smooth, stride_smh, has_smooth = _smoothing_arg(key_smoothing, kv_heads, head_dim, query.device, key_scales)
    _int8_decode_pair_kernel[(batch, kv_heads)](
        q_view, key_pages, value_pages, key_scales, value_scales, o_view, block_tables, seq_lens,
        smooth, stride_smh, *q_view.stride(), *key_pages.stride(), *value_pages.stride(),
        *key_scales.stride(), *value_scales.stride(), *o_view.stride(), *block_tables.stride(),
        seq_lens.stride(0), scale, BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCK_N=block_n,
        LENGTH_OFFSET=length_offset, HAS_SMOOTH=has_smooth, num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def paged_decode_int8_per_head(
    query, key_pages, value_pages, key_scales, value_scales, block_tables, seq_lens, *,
    scale=None, block_n: int = 128, num_warps: int = 4, num_stages: int = 2,
    length_offset: int = 0, key_smoothing=None,
) -> torch.Tensor:
    """The existing per-head INT8 kernel with every launch parameter exposed."""
    from engine.kernels.int8_paged_kv import _paged_decode_batched_int8_kernel, _smoothing_arg

    batch, q_heads, _, head_dim = query.shape
    _, block_size, kv_heads, _ = key_pages.shape
    if scale is None:
        scale = head_dim ** -0.5
    query = query.contiguous()
    output = torch.empty_like(query)
    q_view, o_view = query.view(batch, q_heads, head_dim), output.view(batch, q_heads, head_dim)
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=query.device)
    smooth, stride_smh, has_smooth = _smoothing_arg(key_smoothing, kv_heads, head_dim, query.device, key_scales)
    _paged_decode_batched_int8_kernel[(batch, q_heads)](
        q_view, key_pages, value_pages, key_scales, value_scales, o_view, block_tables, seq_lens,
        smooth, stride_smh, *q_view.stride(), *key_pages.stride(), *value_pages.stride(),
        *key_scales.stride(), *value_scales.stride(), *o_view.stride(), *block_tables.stride(),
        seq_lens.stride(0), q_heads, kv_heads, scale, BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
        BLOCK_N=block_n, LENGTH_OFFSET=length_offset, HAS_SMOOTH=has_smooth,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output
