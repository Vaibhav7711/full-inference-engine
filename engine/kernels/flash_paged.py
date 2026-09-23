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
def flash_dense():
    """The dense ``flash_attn_func`` entry point, if importable."""
    try:
        from flash_attn import flash_attn_func
    except Exception:
        return None
    return flash_attn_func


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
    # `flash_attn_with_kvcache` validates this at launch. Checking it here makes backend
    # resolution honest: the engine's normal 16-token pages cannot use this API, so
    # selecting `auto` must fall through rather than fail halfway through a request.
    if page_size % 256:
        return (
            "flash_attn_with_kvcache needs page sizes divisible by 256, "
            f"not {page_size}"
        )
    if head_dim % 8 or head_dim > 256:
        return f"head_dim {head_dim} is not supported by flash attention"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"flash attention needs fp16 or bf16 activations, not {dtype}"
    return None


def supports_dense(head_dim: int, dtype: torch.dtype) -> str | None:
    """Geometry check for gathered dense FlashAttention (no page-size restriction)."""
    reason = unavailable_reason()
    if reason is not None:
        return reason
    if flash_dense() is None:
        return "flash_attn_func is unavailable"
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
    num_splits: int = 0,
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
        num_splits=num_splits,
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
    # The engine stages every row at its fixed chunk width and uses `chunk_lens` to
    # distinguish real tokens from trailing padding. `flash_attn_with_kvcache` has no
    # query-length argument: it treats every token in `q` as real. Passing the padded
    # width therefore shifts causal alignment for short rows and silently changes
    # greedy tokens. Group by the actual query length, preserving a single batched call
    # for the common equal-length case; rows with different lengths need separate calls.
    batch, _, width, _ = query.shape
    lengths = start_positions.to(torch.int32) + chunk_lens.to(torch.int32)
    tables = block_tables.to(torch.int32)
    groups = cache.get("flash_row_groups") if cache is not None else None
    if groups is None:
        # Standalone callers do not have the engine's host staging. This fallback is
        # correct but synchronizes; the live engine precomputes groups once per step.
        valid_lens = [int(length) for length in chunk_lens.detach().to("cpu").tolist()]
        if len(valid_lens) != batch or any(length <= 0 or length > width for length in valid_lens):
            raise ValueError(f"invalid prefill chunk lengths {valid_lens} for width {width}")
        rows_by_length: dict[int, list[int]] = {}
        for row, length in enumerate(valid_lens):
            rows_by_length.setdefault(length, []).append(row)
        groups = tuple(
            (length, torch.tensor(rows, device=query.device, dtype=torch.long))
            for length, rows in rows_by_length.items()
        )

    # The normal path: every scheduled row has the same real width. Avoid all gathers,
    # padding buffers and scatter copies and hand the engine tensors directly to FA.
    if len(groups) == 1 and groups[0][1] is None:
        length = groups[0][0]
        if not 0 < length <= width:
            raise ValueError(f"invalid prefill chunk length {length} for width {width}")
        result = kernel(
            query[:, :, :length, :].transpose(1, 2), key_pages, value_pages,
            cache_seqlens=lengths, block_table=tables, softmax_scale=scale, causal=True,
        )
        return result.transpose(1, 2).contiguous()

    out = torch.zeros_like(query)
    for length, row_indices in groups:
        if row_indices is None or not 0 < length <= width:
            raise ValueError(f"invalid Flash prefill row group length {length}")
        result = kernel(
            query.index_select(0, row_indices)[:, :, :length, :].transpose(1, 2),
            key_pages, value_pages,
            cache_seqlens=lengths.index_select(0, row_indices),
            block_table=tables.index_select(0, row_indices),
            softmax_scale=scale,
            causal=True,
        ).transpose(1, 2)
        grouped_out = torch.zeros_like(query.index_select(0, row_indices))
        grouped_out[:, :, :length, :] = result
        out.index_copy_(0, row_indices, grouped_out)
    return out


def flash_dense_prefill(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    start_positions: torch.Tensor,
    chunk_lens: torch.Tensor,
    *,
    scale: float | None = None,
    total_len: int = 0,
    cache: dict | None = None,
    **_: object,
) -> torch.Tensor:
    """Gather paged KV, then use dense FA2; supports the engine's 16-token pages.

    FA2's direct paged entry point requires 256-token pages. This variant retains the
    normal allocator geometry and pays the same page gather as SDPA, while avoiding its
    explicit mask and folded-GQA query layout. Rows are grouped by real query and cache
    length so bottom-right causal alignment remains exact.
    """
    kernel = flash_dense()
    if kernel is None:
        raise RuntimeError(unavailable_reason() or "flash_attn_func is unavailable")
    from engine.kernels.sdpa_prefill import gather_pages

    batch, _, width, _ = query.shape
    groups = cache.get("flash_dense_row_groups") if cache is not None else None
    if groups is None:
        starts = start_positions.detach().to("cpu").tolist()
        chunks = chunk_lens.detach().to("cpu").tolist()
        grouped: dict[tuple[int, int], list[int]] = {}
        for row, (start, length) in enumerate(zip(starts, chunks)):
            grouped.setdefault((int(length), int(start + length)), []).append(row)
        groups = tuple(
            (length, length_total, torch.tensor(rows, device=query.device, dtype=torch.long))
            for (length, length_total), rows in grouped.items() if length > 0
        )

    out = torch.zeros_like(query)
    for length, length_total, row_indices in groups:
        if not 0 < length <= width or length_total < length:
            raise ValueError(f"invalid dense Flash group q={length}, kv={length_total}")
        if row_indices is None:
            q = query[:, :, :length, :].transpose(1, 2)
            tables = block_tables
        else:
            q = query.index_select(0, row_indices)[:, :, :length, :].transpose(1, 2)
            tables = block_tables.index_select(0, row_indices)
        keys = gather_pages(key_pages, tables, length_total).transpose(1, 2)
        values = gather_pages(value_pages, tables, length_total).transpose(1, 2)
        result = kernel(q, keys, values, softmax_scale=scale, causal=True).transpose(1, 2)
        if row_indices is None:
            return result.contiguous()
        padded = torch.zeros(
            (len(row_indices), query.shape[1], width, query.shape[3]),
            device=query.device, dtype=query.dtype,
        )
        padded[:, :, :length, :] = result
        out.index_copy_(0, row_indices, padded)
    return out
