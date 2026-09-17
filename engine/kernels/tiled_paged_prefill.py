"""Tiled causal prefill attention over the shared paged KV pool.

Replaces the per-query-token kernel in `paged_prefill.py`, which launched one program per
(row, head, query token) and had each program walk the whole KV prefix from position zero.
For a 128-token chunk with 16 query heads that is 2,048 programs re-reading the same keys
and values: a measured 28.7 GB of KV traffic per prefill step across 28 layers, against
0.46 GB if each tile of queries loads the KV it needs once. It also scored with
`tl.sum(q * k, axis=-1)`, an elementwise multiply and reduction on CUDA cores, leaving the
tensor cores idle.

This kernel is the standard FlashAttention-2 structure adapted to paged KV:

- one program per (row, head, tile of BLOCK_M query tokens)
- KV streamed in BLOCK_N tiles, gathered through the block table
- `tl.dot` for both QK^T and PV, so scoring runs on tensor cores
- online softmax, fp32 accumulator
- causal bound per tile: a tile only visits KV up to the last position its rows can see

The paged part is the one real deviation from a contiguous FlashAttention: keys and values
for logical position `p` live at `block_table[p // BLOCK_SIZE]`, offset `p % BLOCK_SIZE`,
so every KV tile load is a gather rather than a strided read.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Masked scores use a large finite sentinel rather than -inf. A padded query row has every
# score masked, and -inf would make (m_i - m_new) evaluate to (-inf) - (-inf) = NaN in the
# online-softmax rescale. With a finite sentinel the rescale is exp(0) = 1 and the row's
# probabilities are forced to exact zeros below.
_NEG = -1.0e30


@triton.jit
def _tiled_paged_prefill_kernel(
    q_ptr, kp_ptr, vp_ptr, out_ptr, bt_ptr, starts_ptr, chunks_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_btb, stride_btl,
    num_q_heads, num_kv_heads, scale, max_blocks, num_pool_blocks,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PV_IN_FP32: tl.constexpr,
):
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    tile = tl.program_id(2)

    chunk_len = tl.load(chunks_ptr + batch)
    start = tl.load(starts_ptr + batch)

    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    row_valid = offs_m < chunk_len
    # Absolute position of each query row in the sequence: the chunk starts at `start`.
    q_pos = start + offs_m
    kv_head = q_head // (num_q_heads // num_kv_heads)

    q_base = q_ptr + batch * stride_qb + q_head * stride_qh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
        mask=row_valid[:, None], other=0.0,
    )

    m_i = tl.full([BLOCK_M], _NEG, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal bound for the whole tile: no row here can see past the last row's position.
    # Tiles early in the chunk therefore stream far less KV than tiles late in it.
    tile_last = tl.minimum(tile * BLOCK_M + BLOCK_M, chunk_len)
    kv_end = start + tile_last
    table = bt_ptr + batch * stride_btb

    for start_n in range(0, kv_end, BLOCK_N):
        positions = start_n + offs_n
        in_range = positions < kv_end
        logical = positions // BLOCK_SIZE
        offsets = positions % BLOCK_SIZE
        # Bounds-checked gather: a stale or out-of-range table entry must not read another
        # request's pages, matching the guarantee the KV write kernels make.
        table_ok = in_range & (logical < max_blocks)
        physical = tl.load(table + logical * stride_btl, mask=table_ok, other=-1)
        valid_n = table_ok & (physical >= 0) & (physical < num_pool_blocks)

        kv_offset = (
            physical[:, None] * stride_kb + offsets[:, None] * stride_ks
            + kv_head * stride_kh + offs_d[None, :] * stride_kd
        )
        keys = tl.load(kp_ptr + kv_offset, mask=valid_n[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(keys)) * scale
        visible = (
            (positions[None, :] <= q_pos[:, None])
            & valid_n[None, :] & row_valid[:, None]
        )
        scores = tl.where(visible, scores, _NEG)

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - new_max)
        probs = tl.exp(scores - new_max[:, None])
        # Force exact zeros: with a finite sentinel the masked entries would otherwise
        # carry exp(_NEG - new_max), which is tiny but not zero, and a fully masked row
        # would accumulate weight it should not have.
        probs = tl.where(visible, probs, 0.0)

        v_offset = (
            physical[:, None] * stride_vb + offsets[:, None] * stride_vs
            + kv_head * stride_vh + offs_d[None, :] * stride_vd
        )
        values = tl.load(vp_ptr + v_offset, mask=valid_n[:, None], other=0.0)

        acc = acc * alpha[:, None]
        if PV_IN_FP32:
            # Matches the reference kernel's fp32 PV accumulation exactly, at the cost of
            # running this matmul on CUDA cores instead of tensor cores.
            acc += tl.sum(probs[:, :, None] * values.to(tl.float32)[None, :, :], axis=1)
        else:
            acc += tl.dot(probs.to(values.dtype), values)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = new_max

    result = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    result = tl.where(row_valid[:, None], result, 0.0)
    out_base = out_ptr + batch * stride_ob + q_head * stride_oh
    tl.store(
        out_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
        result.to(out_ptr.dtype.element_ty), mask=row_valid[:, None],
    )


def tiled_paged_prefill(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    start_positions: torch.Tensor,
    chunk_lens: torch.Tensor,
    *,
    scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    pv_in_fp32: bool = False,
) -> torch.Tensor:
    """Attend each padded query chunk to its paged prefix plus its own causal region.

    Drop-in replacement for `paged_prefill` with the same signature and semantics.
    """
    if query.ndim != 4:
        raise ValueError("query must have shape [B,H,Q,D]")
    batch, q_heads, query_len, head_dim = query.shape
    if key_pages.shape != value_pages.shape or key_pages.ndim != 4:
        raise ValueError("K/V pools must have matching [blocks,block,H,D] shapes")
    _, block_size, kv_heads, kv_dim = key_pages.shape
    if kv_dim != head_dim or head_dim > 128 or q_heads % kv_heads:
        raise ValueError("unsupported attention head geometry")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError("block tables must have shape [B,max_blocks]")
    if start_positions.shape != (batch,) or chunk_lens.shape != (batch,):
        raise ValueError("start positions and chunk lengths must have shape [B]")
    if block_m < 16 or block_n < 16:
        raise ValueError("tl.dot requires tiles of at least 16 in each dimension")
    if scale is None:
        scale = head_dim ** -0.5

    query = query.contiguous()
    key_pages = key_pages.contiguous()
    value_pages = value_pages.contiguous()
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    start_positions = start_positions.contiguous().to(dtype=torch.int32, device=query.device)
    chunk_lens = chunk_lens.contiguous().to(dtype=torch.int32, device=query.device)
    out = torch.empty_like(query)
    grid = (batch, q_heads, triton.cdiv(query_len, block_m))
    _tiled_paged_prefill_kernel[grid](
        query, key_pages, value_pages, out, block_tables, start_positions, chunk_lens,
        *query.stride(), *key_pages.stride(), *value_pages.stride(), *out.stride(),
        *block_tables.stride(), q_heads, kv_heads, scale,
        block_tables.shape[1], key_pages.shape[0],
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
        BLOCK_M=block_m, BLOCK_N=block_n, PV_IN_FP32=pv_in_fp32,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
