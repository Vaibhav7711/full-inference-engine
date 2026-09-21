"""Paged decode attention with the K/V tile shared across a GQA group.

`paged_decode_batched` runs one program per (sequence, query head). Under grouped-query
attention the `n_rep` query heads of a group read the same KV head, so every K and V
tile is streamed from the pool `n_rep` times per layer - twice for Qwen3-0.6B. Decode is
the memory-bound half of serving and, at batch 8-16 with ~700-token contexts, KV traffic
is of the same order as the weight read the step is floored by.

This variant runs one program per (sequence, KV head) with the group's two query heads
unrolled: two sets of online-softmax state, one K tile and one V tile per iteration
feeding both. All products are rank-2, as in the per-head kernel. (A first version
carried the group as a tensor axis and broadcast to `[REP, BLOCK_N, D]`; it lost 1.5-2.8x
to the per-head kernel at every point of the T4 sweep, so the layout, not the shared
read, was what got measured. Journal, Phase 2c.) Written for `n_rep == 2`, Qwen3-0.6B's
geometry; other group sizes use the per-head kernel.

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
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LENGTH_OFFSET: tl.constexpr,
):
    """One program per (sequence, KV head), the group's two query heads unrolled.

    Every product is rank-2 (`[BLOCK_N, D]`), exactly as in the per-head kernel; the
    only difference is that one K tile and one V tile feed two heads. The first version
    of this kernel carried the group as a leading tensor axis and broadcast to
    `[REP, BLOCK_N, D]`; that ran 1.5-2.8x slower than the per-head kernel on the T4 at
    every operating point, including ones with ample parallelism, so the rank-3 layout
    and not the halved grid was the cost. This form isolates the shared read.
    """
    pid_s = tl.program_id(0)
    pid_kv = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    seq_len = tl.load(sl_ptr + pid_s * stride_sl) + LENGTH_OFFSET

    h0 = pid_kv * 2
    q_base = q_ptr + pid_s * stride_qs + offs_d * stride_qd
    q0 = tl.load(q_base + h0 * stride_qh)
    q1 = tl.load(q_base + (h0 + 1) * stride_qh)

    m0 = float("-inf")
    l0 = 0.0
    acc0 = tl.zeros([HEAD_DIM], dtype=tl.float32)
    m1 = float("-inf")
    l1 = 0.0
    acc1 = tl.zeros([HEAD_DIM], dtype=tl.float32)

    bt_base = bt_ptr + pid_s * stride_bts
    for start_n in range(0, seq_len, BLOCK_N):
        cur_n = start_n + offs_n
        n_mask = cur_n < seq_len
        logical_block = cur_n // BLOCK_SIZE
        offset = cur_n % BLOCK_SIZE
        phys_block = tl.load(bt_base + logical_block * stride_btb, mask=n_mask, other=0)

        k_row = phys_block[:, None] * stride_kb + offset[:, None] * stride_ks + pid_kv * stride_kh
        k = tl.load(kp_ptr + k_row + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)                       # [BLOCK_N, D], once

        s0 = tl.where(n_mask, tl.sum(q0[None, :] * k, axis=1) * scale, float("-inf"))
        s1 = tl.where(n_mask, tl.sum(q1[None, :] * k, axis=1) * scale, float("-inf"))
        mn0 = tl.maximum(m0, tl.max(s0, axis=0))
        mn1 = tl.maximum(m1, tl.max(s1, axis=0))
        a0 = tl.exp(m0 - mn0)
        a1 = tl.exp(m1 - mn1)
        p0 = tl.exp(s0 - mn0)
        p1 = tl.exp(s1 - mn1)

        v_row = phys_block[:, None] * stride_vb + offset[:, None] * stride_vs + pid_kv * stride_vh
        v = tl.load(vp_ptr + v_row + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)                       # [BLOCK_N, D], once

        acc0 = acc0 * a0 + tl.sum(p0[:, None] * v, axis=0)
        acc1 = acc1 * a1 + tl.sum(p1[:, None] * v, axis=0)
        l0 = l0 * a0 + tl.sum(p0, axis=0)
        l1 = l1 * a1 + tl.sum(p1, axis=0)
        m0 = mn0
        m1 = mn1

    o_base = out_ptr + pid_s * stride_os + offs_d * stride_od
    tl.store(o_base + h0 * stride_oh, (acc0 / tl.where(l0 > 0.0, l0, 1.0)).to(out_ptr.dtype.element_ty))
    tl.store(o_base + (h0 + 1) * stride_oh, (acc1 / tl.where(l1 > 0.0, l1, 1.0)).to(out_ptr.dtype.element_ty))


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
    if H // kv_heads != 2:
        raise ValueError("the shared kernel is written for a GQA group of exactly 2 query heads")
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
        BLOCK_SIZE=block_size, HEAD_DIM=D, BLOCK_N=block_n,
        LENGTH_OFFSET=length_offset, num_warps=num_warps,
    )
    return out
