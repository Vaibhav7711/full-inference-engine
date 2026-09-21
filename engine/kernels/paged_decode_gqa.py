"""Paged decode attention with the K/V tile shared across a GQA group.

`paged_decode_batched` runs one program per (sequence, query head). Under grouped-query
attention the `n_rep` query heads of a group read the same KV head, so every K and V
tile is streamed from the pool `n_rep` times per layer - twice for Qwen3-0.6B. Decode is
the memory-bound half of serving and, at batch 8-16 with ~700-token contexts, KV traffic
is of the same order as the weight read the step is floored by.

This variant runs one program per (sequence, KV head) and carries the online-softmax
state of all `n_rep` query heads: `[REP]` running max and sum, `[REP, D]` accumulator.
Each K/V tile is loaded once and used for every head in the group. The score and PV
products broadcast to `[REP, BLOCK_N, D]` in registers, so a tile of BLOCK_N here costs
what a tile of REP * BLOCK_N costs in the per-head kernel; the regime table halves
BLOCK_N accordingly.

What it does not fix: the grid shrinks by `n_rep` (S * kv_heads programs), so at small
batches fewer SMs are busy. Whether the halved traffic beats the lost parallelism on a
given GPU is what `benchmarks/kernels/paged_decode_regime_sweep.py --gqa` measures; the
engine selects the variant with `decode_attention`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_gqa_kernel(
    q_ptr, kp_ptr, vp_ptr, out_ptr, bt_ptr, sl_ptr,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_os, stride_oh, stride_od,
    stride_bts, stride_btb,
    stride_sl,
    scale,
    REP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LENGTH_OFFSET: tl.constexpr,
):
    pid_s = tl.program_id(0)    # sequence
    pid_kv = tl.program_id(1)   # KV head; query heads kv*REP .. kv*REP+REP-1

    offs_r = tl.arange(0, REP)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    seq_len = tl.load(sl_ptr + pid_s * stride_sl) + LENGTH_OFFSET

    # The group's query vectors: [REP, HEAD_DIM]
    q_heads = pid_kv * REP + offs_r
    q_ptrs = q_ptr + pid_s * stride_qs + q_heads[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)

    m_i = tl.full([REP], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([REP], dtype=tl.float32)
    acc = tl.zeros([REP, HEAD_DIM], dtype=tl.float32)

    bt_base = bt_ptr + pid_s * stride_bts
    kv_off = pid_kv * stride_kh
    v_kv_off = pid_kv * stride_vh

    for start_n in range(0, seq_len, BLOCK_N):
        cur_n = start_n + offs_n
        n_mask = cur_n < seq_len
        logical_block = cur_n // BLOCK_SIZE
        offset = cur_n % BLOCK_SIZE
        phys_block = tl.load(bt_base + logical_block * stride_btb, mask=n_mask, other=0)

        k_row = phys_block[:, None] * stride_kb + offset[:, None] * stride_ks + kv_off
        k = tl.load(kp_ptr + k_row + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)                       # [BLOCK_N, D]

        # scores[r, n] = q[r] . k[n]  -> [REP, BLOCK_N], one tile read for all REP heads
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        m_ij = tl.max(scores, axis=1)                                     # [REP]
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])                               # [REP, BLOCK_N]

        v_row = phys_block[:, None] * stride_vb + offset[:, None] * stride_vs + v_kv_off
        v = tl.load(vp_ptr + v_row + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)                       # [BLOCK_N, D]

        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    o_ptrs = out_ptr + pid_s * stride_os + q_heads[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty))


def paged_decode_gqa(
    query: torch.Tensor,        # [S, H, 1, D]
    key_pages: torch.Tensor,    # [num_blocks, block_size, kv_heads, D]
    value_pages: torch.Tensor,
    block_tables: torch.Tensor, # [S, max_blocks] int
    seq_lens: torch.Tensor,     # [S] int
    scale: float | None = None,
    block_n: int = 32,
    num_warps: int = 4,
    length_offset: int = 0,
) -> torch.Tensor:
    """Same contract as `paged_decode_batched`; one program per (sequence, KV head)."""
    S, H, one, D = query.shape
    if one != 1:
        raise ValueError("decode: query length must be 1 per sequence")
    num_blocks, block_size, kv_heads, D_kv = key_pages.shape
    if D_kv != D or D > 128:
        raise ValueError("head_dim mismatch or above 128")
    if H % kv_heads:
        raise ValueError("num_q_heads must be a multiple of kv_heads")
    rep = H // kv_heads
    if rep & (rep - 1):
        raise ValueError("GQA group size must be a power of two for the shared kernel")
    if block_n not in {16, 32, 64, 128}:
        raise ValueError("block_n must be one of 16, 32, 64, or 128")
    if num_warps not in {2, 4, 8}:
        raise ValueError("num_warps must be one of 2, 4, or 8")
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    query = query.contiguous()
    key_pages = key_pages.contiguous()
    value_pages = value_pages.contiguous()
    block_tables = block_tables.contiguous().to(torch.int32)
    seq_lens = seq_lens.contiguous().to(torch.int32)
    out = torch.empty_like(query)
    q_v = query.view(S, H, D)
    o_v = out.view(S, H, D)

    _paged_decode_gqa_kernel[(S, kv_heads)](
        q_v, key_pages, value_pages, o_v, block_tables, seq_lens,
        q_v.stride(0), q_v.stride(1), q_v.stride(2),
        key_pages.stride(0), key_pages.stride(1), key_pages.stride(2), key_pages.stride(3),
        value_pages.stride(0), value_pages.stride(1), value_pages.stride(2), value_pages.stride(3),
        o_v.stride(0), o_v.stride(1), o_v.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        seq_lens.stride(0),
        scale,
        REP=rep, BLOCK_SIZE=block_size, HEAD_DIM=D, BLOCK_N=block_n,
        LENGTH_OFFSET=length_offset, num_warps=num_warps,
    )
    return out
