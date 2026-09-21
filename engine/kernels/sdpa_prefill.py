"""Chunked prefill attention through torch SDPA over gathered pages.

The benchmark in `benchmarks/kernels/prefill_attention_ab.py` used exactly this as its
"upper bound": gather the prefix pages into a dense tensor, let torch attend. On the T4 it
is not merely a bound but the fastest correct implementation available, because Triton's
`tl.dot` does not reach the tensor cores on sm_75 (see the journal entry "The tiled prefill
kernel never used the tensor cores") while PyTorch's memory-efficient SDPA kernel does.

Cost of the gather: one copy of the prefix K and V per layer per chunk step - at a 896-token
prefix, 3.7 MB per row per layer, about 1.6 ms across 28 layers at the measured 258 GB/s.
The tiled Triton kernel spent 37 ms per layer on the same shape.

It is also the same SDPA kernel the fresh-prompt fast path runs, so a prompt prefilled in
chunks and a prompt prefilled whole go through one attention implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def chunk_causal_mask(
    start_positions: torch.Tensor,
    chunk_lens: torch.Tensor,
    query_len: int,
    total_len: int,
) -> torch.Tensor:
    """Boolean `[B, 1, query_len, total_len]` mask for a padded batch of prefill chunks.

    Row `j` of batch `b` is the token at absolute position `start[b] + j`; it may see every
    key at position `p <= start[b] + j`. Rows past `chunk_lens[b]` are padding: they are
    given key 0 only, so softmax has one finite entry and produces no NaN, and their
    output is never read.
    """
    device = start_positions.device
    positions = torch.arange(total_len, device=device)
    rows = torch.arange(query_len, device=device)
    absolute = start_positions[:, None].to(torch.long) + rows[None, :]          # [B, Q]
    visible = positions[None, None, :] <= absolute[:, :, None]                  # [B, Q, T]
    row_valid = rows[None, :] < chunk_lens[:, None].to(torch.long)              # [B, Q]
    padding_rows = ~row_valid[:, :, None]
    first_key_only = positions[None, None, :] == 0
    visible = torch.where(padding_rows, first_key_only, visible)
    return visible[:, None, :, :]


def gather_pages(
    pool: torch.Tensor, block_tables: torch.Tensor, total_len: int,
) -> torch.Tensor:
    """Dense `[B, H, total_len, D]` view of each row's first `total_len` logical tokens.

    Block-table entries past a row's allocation are `-1`; they are clamped to page 0 and
    the positions they cover are never visible under the causal bound, since every row's
    visible keys end at its own `start + chunk`.
    """
    batch = block_tables.shape[0]
    block_size = pool.shape[1]
    blocks = -(-total_len // block_size)
    pages = block_tables[:, :blocks].clamp(min=0)
    gathered = pool.index_select(0, pages.reshape(-1).to(torch.long))          # [B*blocks, S, H, D]
    gathered = gathered.view(batch, blocks * block_size, *pool.shape[2:])[:, :total_len]
    return gathered.permute(0, 2, 1, 3)                                         # [B, H, T, D]


def sdpa_paged_prefill(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    start_positions: torch.Tensor,
    chunk_lens: torch.Tensor,
    *,
    total_len: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Attend each padded query chunk to its paged prefix plus its own causal region.

    Same signature and semantics as `paged_prefill`, plus `total_len`: the longest
    `start + chunk` in the batch, which the caller knows host-side at planning time (it
    must not be read back from the device tensors, that would be a sync per layer).
    """
    if query.ndim != 4:
        raise ValueError("query must have shape [B,H,Q,D]")
    batch, q_heads, query_len, head_dim = query.shape
    kv_heads = key_pages.shape[2]
    if q_heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    if total_len <= 0:
        raise ValueError("total_len must be positive")
    keys = gather_pages(key_pages, block_tables, total_len)
    values = gather_pages(value_pages, block_tables, total_len)
    mask = chunk_causal_mask(start_positions, chunk_lens, query_len, total_len)
    if scale is None:
        scale = head_dim ** -0.5
    repeat = q_heads // kv_heads
    if repeat == 1:
        return F.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=scale)
    # GQA without `enable_gqa`: that flag is honoured only by the math backend on this
    # torch/GPU (no flash on sm_75, and the memory-efficient kernel rejects it), which
    # materialises fp32 score matrices and cost +37% prefill GPU time on the T4. Instead
    # fold the `repeat` query heads that share a KV head into the query-length axis, so
    # the kernel sees [B, kv_heads, repeat*Q, D] against [B, kv_heads, T, D] and the
    # memory-efficient path applies. Query head h = kv * repeat + r, matching the Triton
    # kernels' `h // repeat` mapping.
    grouped = query.reshape(batch, kv_heads, repeat * query_len, head_dim)
    grouped_mask = mask.expand(batch, repeat, query_len, total_len).reshape(
        batch, 1, repeat * query_len, total_len,
    )
    out = F.scaled_dot_product_attention(grouped, keys, values, attn_mask=grouped_mask, scale=scale)
    return out.reshape(batch, q_heads, query_len, head_dim)
