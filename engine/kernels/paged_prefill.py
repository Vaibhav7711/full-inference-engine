"""Chunked causal prefill attention over the shared paged KV pool."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_prefill_kernel(
    q_ptr, kp_ptr, vp_ptr, out_ptr, bt_ptr, starts_ptr, chunks_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_btb, stride_btl,
    num_q_heads, num_kv_heads, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    q_head = tl.program_id(1)
    query_token = tl.program_id(2)
    dims = tl.arange(0, HEAD_DIM)

    chunk_len = tl.load(chunks_ptr + batch)
    query_valid = query_token < chunk_len
    start = tl.load(starts_ptr + batch)
    kv_len = tl.where(query_valid, start + query_token + 1, 0)
    kv_head = q_head // (num_q_heads // num_kv_heads)

    q_base = (
        q_ptr + batch * stride_qb + q_head * stride_qh
        + query_token * stride_qt
    )
    q = tl.load(q_base + dims * stride_qd, mask=query_valid, other=0.0)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    offsets_n = tl.arange(0, BLOCK_N)
    table = bt_ptr + batch * stride_btb

    for start_n in range(0, kv_len, BLOCK_N):
        positions = start_n + offsets_n
        mask_n = positions < kv_len
        logical_blocks = positions // BLOCK_SIZE
        block_offsets = positions % BLOCK_SIZE
        physical_blocks = tl.load(
            table + logical_blocks * stride_btl, mask=mask_n, other=0,
        )
        k_rows = (
            physical_blocks[:, None] * stride_kb
            + block_offsets[:, None] * stride_ks + kv_head * stride_kh
        )
        keys = tl.load(
            kp_ptr + k_rows + dims[None, :] * stride_kd,
            mask=mask_n[:, None], other=0.0,
        )
        scores = tl.sum(q[None, :] * keys, axis=1) * scale
        scores = tl.where(mask_n, scores, float("-inf"))
        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - new_max)
        probabilities = tl.exp(scores - new_max)
        v_rows = (
            physical_blocks[:, None] * stride_vb
            + block_offsets[:, None] * stride_vs + kv_head * stride_vh
        )
        values = tl.load(
            vp_ptr + v_rows + dims[None, :] * stride_vd,
            mask=mask_n[:, None], other=0.0,
        )
        acc = acc * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        l_i = l_i * alpha + tl.sum(probabilities, axis=0)
        m_i = new_max

    result = tl.where(query_valid, acc / tl.where(l_i > 0.0, l_i, 1.0), 0.0)
    out_base = (
        out_ptr + batch * stride_ob + q_head * stride_oh
        + query_token * stride_ot
    )
    tl.store(out_base + dims * stride_od, result.to(out_ptr.dtype.element_ty))


def paged_prefill(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    start_positions: torch.Tensor,
    chunk_lens: torch.Tensor,
    *,
    scale: float | None = None,
    block_n: int = 64,
) -> torch.Tensor:
    """Attend each padded query chunk to its paged prefix plus causal chunk."""
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
    if scale is None:
        scale = head_dim ** -0.5

    query = query.contiguous()
    key_pages = key_pages.contiguous()
    value_pages = value_pages.contiguous()
    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=query.device)
    start_positions = start_positions.contiguous().to(dtype=torch.int32, device=query.device)
    chunk_lens = chunk_lens.contiguous().to(dtype=torch.int32, device=query.device)
    out = torch.empty_like(query)
    _paged_prefill_kernel[(batch, q_heads, query_len)](
        query, key_pages, value_pages, out, block_tables, start_positions, chunk_lens,
        *query.stride(), *key_pages.stride(), *value_pages.stride(), *out.stride(),
        *block_tables.stride(), q_heads, kv_heads, scale,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCK_N=block_n,
        num_warps=4,
    )
    return out
