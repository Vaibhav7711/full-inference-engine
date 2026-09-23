"""FlashAttention-2 over this engine's paged KV pool, when the wheel is installed.

`flash_attn_with_kvcache` takes exactly the layout the engine already stores - a pool of
`[num_blocks, page_size, kv_heads, head_dim]` pages plus an `int32` block table and a
per-row cache length - and handles GQA natively, splits the key range internally for
decode (FlashDecoding), and needs no mask tensor for the causal chunk case. On a GPU
where it is available it is the kernel to beat: it is what production stacks serve with,
and it removes both the page gather the SDPA path pays and the mask the chunk path
builds.

It is not available everywhere. The wheel is built per architecture and the released
ones start at sm_80, which is why this module is import-guarded and the backend registry
treats its absence as a reason rather than an error: the T4 results were produced without
it and must stay reproducible.

Nothing here computes attention itself; it maps the engine's tensors onto the flash
signature and back.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def flash_kvcache():
    """`flash_attn_with_kvcache` if importable here, else None."""
    try:
        from flash_attn import flash_attn_with_kvcache
    except Exception:
        return None
    return flash_attn_with_kvcache


@lru_cache(maxsize=1)
def unavailable_reason() -> str | None:
    """Why FlashAttention cannot be used on this machine, or None if it can."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if flash_kvcache() is None:
        return "flash_attn is not installed (pip install flash-attn, sm_80+ wheels)"
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        return f"flash_attn wheels require sm_80+; this device is sm_{major}x"
    return None


def supports(page_size: int, head_dim: int, dtype: torch.dtype) -> str | None:
    """Geometry-level check, separate from the import check."""
    if page_size % 16:
        return f"paged flash attention needs a page size that is a multiple of 16, not {page_size}"
    if head_dim % 8 or head_dim > 256:
        return f"head_dim {head_dim} is not supported by flash attention"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"flash attention needs fp16 or bf16 activations, not {dtype}"
    return None


def flash_paged_decode(
    query: torch.Tensor,        # [S, H, 1, D]
    key_pages: torch.Tensor,    # [num_blocks, page_size, kv_heads, D]
    value_pages: torch.Tensor,
    block_tables: torch.Tensor, # [S, max_blocks] int32
    seq_lens: torch.Tensor,     # [S] int32, KV length BEFORE this step's token
    scale: float | None = None,
    block_n: int = 64,          # accepted for a uniform backend signature; unused
    num_warps: int = 4,         # likewise
    length_offset: int = 0,
    **_: object,
) -> torch.Tensor:
    """One decode step. The K/V for this step must already be in the pool.

    The engine's own write kernel stores it before attention is called, so `cache_seqlens`
    is `seq_lens + length_offset`: the row's history *including* the token just written.
    """
    kernel = flash_kvcache()
    if kernel is None:
        raise RuntimeError(unavailable_reason() or "flash_attn is unavailable")
    S, H, one, D = query.shape
    if one != 1:
        raise ValueError("decode: query length must be 1 per sequence")
    lengths = seq_lens.to(torch.int32)
    if length_offset:
        lengths = lengths + length_offset
    out = kernel(
        query.transpose(1, 2),                      # [S, 1, H, D]
        key_pages, value_pages,
        cache_seqlens=lengths,
        block_table=block_tables.to(torch.int32),
        softmax_scale=scale,
        causal=False,                               # one query token sees the whole prefix
    )
    return out.transpose(1, 2).contiguous()         # back to [S, H, 1, D]


def flash_paged_prefill(
    query: torch.Tensor,        # [B, H, Q, D]
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    start_positions: torch.Tensor,  # [B] int32, absolute position of each row's chunk
    chunk_lens: torch.Tensor,       # [B] int32
    *,
    scale: float | None = None,
    total_len: int = 0,
    cache: dict | None = None,
    **_: object,
) -> torch.Tensor:
    """One chunked prefill step, causal within the chunk and over the paged prefix.

    `cache_seqlens = start + chunk` is each row's history including this chunk, which the
    write kernel has already stored; with `causal=True` flash aligns the query block to
    the end of that history, which is exactly the chunk's own causal region.
    """
    kernel = flash_kvcache()
    if kernel is None:
        raise RuntimeError(unavailable_reason() or "flash_attn is unavailable")
    lengths = (start_positions.to(torch.int32) + chunk_lens.to(torch.int32))
    out = kernel(
        query.transpose(1, 2),                      # [B, Q, H, D]
        key_pages, value_pages,
        cache_seqlens=lengths,
        block_table=block_tables.to(torch.int32),
        softmax_scale=scale,
        causal=True,
    )
    return out.transpose(1, 2).contiguous()
