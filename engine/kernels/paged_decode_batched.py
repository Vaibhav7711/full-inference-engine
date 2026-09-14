"""K4: multi-sequence batched paged attention (decode) — one kernel launch, S sequences.

This is what continuous batching needs: process S sequences TOGETHER in one attention
launch, each with its own length and its own physical blocks, instead of S separate
single-sequence launches.

Built on K1 (math) + K2 (paged addressing). K4 adds the multi-sequence dimension for
the DECODE case: each sequence contributes exactly one query token attending to its own
full KV history. That's the continuous-batching workload — many sequences each emitting
one token per step.

Decode layout:
    query:        [S, H, 1, D]                          S sequences, 1 query token each
    key_pages:    [num_blocks, block_size, kv_heads, D]  ONE shared block pool
    value_pages:  [num_blocks, block_size, kv_heads, D]
    block_tables: [S, max_blocks]  int32                 per-sequence logical->physical
    seq_lens:     [S]              int32                  per-sequence KV length
    out:          [S, H, 1, D]

Grid: one program per (sequence, head). Each program loops over ITS sequence's KV using
ITS block table and seq_len — online softmax over that sequence's keys only.

Because M=1 per sequence, no q-tiling: each program handles one query vector. This is
substantially simpler than the general prefill case and matches how paged decode kernels
(vLLM) are structured.

Correctness gate: for each sequence, K4 batched output == K2 single-sequence output.
Transitively == SDPA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_batched_kernel(
    q_ptr, kp_ptr, vp_ptr, out_ptr, bt_ptr, sl_ptr,
    # Q strides [S, H, 1, D]  (query length is 1)
    stride_qs, stride_qh, stride_qd,
    # paged K,V strides [num_blocks, block_size, kv_heads, D]
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    # out strides [S, H, 1, D]
    stride_os, stride_oh, stride_od,
    # block_tables strides [S, max_blocks]
    stride_bts, stride_btb,
    # seq_lens stride [S]
    stride_sl,
    num_q_heads, num_kv_heads,
    scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LENGTH_OFFSET: tl.constexpr,
):
    pid_s = tl.program_id(0)   # which sequence
    pid_h = tl.program_id(1)   # which query head

    # GQA: which KV head does this query head map to?
    n_rep = num_q_heads // num_kv_heads
    kv_head = pid_h // n_rep

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # This sequence's KV length
    seq_len = tl.load(sl_ptr + pid_s * stride_sl) + LENGTH_OFFSET

    # Load this program's single query vector: [HEAD_DIM]
    q_base = q_ptr + pid_s * stride_qs + pid_h * stride_qh
    q = tl.load(q_base + offs_d * stride_qd)     # [HEAD_DIM]

    # Online softmax state (scalars per this one query row)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Block-table base for this sequence
    bt_base = bt_ptr + pid_s * stride_bts

    for start_n in range(0, seq_len, BLOCK_N):
        cur_n = start_n + offs_n                  # [BLOCK_N] logical KV positions
        n_mask = cur_n < seq_len

        # Paged addressing: physical row for each KV position of THIS sequence
        logical_block = cur_n // BLOCK_SIZE
        offset = cur_n % BLOCK_SIZE
        phys_block = tl.load(bt_base + logical_block * stride_btb, mask=n_mask, other=0)

        # K tile: [BLOCK_N, HEAD_DIM] from the shared pool at kv_head
        k_row = phys_block[:, None] * stride_kb + offset[:, None] * stride_ks + kv_head * stride_kh
        k_ptrs = kp_ptr + k_row + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)     # [BLOCK_N, D]

        # scores = q . k^T * scale  -> [BLOCK_N]
        scores = tl.sum(q[None, :] * k, axis=1) * scale          # [BLOCK_N]
        scores = tl.where(n_mask, scores, float("-inf"))
        # Decode: the single query attends to ALL keys of its sequence (no causal mask
        # needed — the query is the newest token, positioned after all cached keys).

        # Online softmax update (reduce over this tile)
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)                                # [BLOCK_N]

        # V tile: [BLOCK_N, HEAD_DIM]
        v_row = phys_block[:, None] * stride_vb + offset[:, None] * stride_vs + kv_head * stride_vh
        v_ptrs = vp_ptr + v_row + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)      # [BLOCK_N, D]

        # acc = acc*alpha + sum_n p[n] * v[n, :]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)        # [HEAD_DIM]
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe

    o_base = out_ptr + pid_s * stride_os + pid_h * stride_oh
    tl.store(o_base + offs_d * stride_od, acc.to(out_ptr.dtype.element_ty))


def paged_decode_batched(
    query: torch.Tensor,        # [S, H, 1, D]
    key_pages: torch.Tensor,    # [num_blocks, block_size, kv_heads, D]
    value_pages: torch.Tensor,  # [num_blocks, block_size, kv_heads, D]
    block_tables: torch.Tensor, # [S, max_blocks] int
    seq_lens: torch.Tensor,     # [S] int
    scale: float | None = None,
    block_n: int = 64,
    length_offset: int = 0,
) -> torch.Tensor:
    """Batched paged decode attention: S sequences, 1 query each, one kernel launch.

    Returns out: [S, H, 1, D]. `length_offset` is applied inside the kernel,
    avoiding a separate elementwise launch when callers store pre-write lengths.
    """
    S, H, one, D = query.shape
    assert one == 1, "K4 decode: query length must be 1 per sequence"
    num_blocks, block_size, kv_heads, D_kv = key_pages.shape
    assert D_kv == D and D <= 128
    assert H % kv_heads == 0, "num_q_heads must be a multiple of kv_heads (GQA)"

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    query = query.contiguous()
    key_pages = key_pages.contiguous()
    value_pages = value_pages.contiguous()
    block_tables = block_tables.contiguous().to(torch.int32)
    seq_lens = seq_lens.contiguous().to(torch.int32)
    out = torch.empty_like(query)

    # Query strides for [S, H, 1, D] — drop the singleton query-length dim in addressing
    q_v = query.view(S, H, D)      # [S, H, D]
    o_v = out.view(S, H, D)

    grid = (S, H)

    _paged_decode_batched_kernel[grid](
        q_v, key_pages, value_pages, o_v, block_tables, seq_lens,
        q_v.stride(0), q_v.stride(1), q_v.stride(2),
        key_pages.stride(0), key_pages.stride(1), key_pages.stride(2), key_pages.stride(3),
        value_pages.stride(0), value_pages.stride(1), value_pages.stride(2), value_pages.stride(3),
        o_v.stride(0), o_v.stride(1), o_v.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        seq_lens.stride(0),
        H, kv_heads,
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        BLOCK_N=block_n,
        LENGTH_OFFSET=length_offset,
    )
    return out
