"""Measured Ada dispatcher for paged decode.

The RTX 4060 sweep found two regimes: a lower-warp per-head launch for general decode,
and paired-GQA sharing for short contexts once enough rows provide parallelism.
"""

from __future__ import annotations

import torch

from engine.kernels.paged_decode_config import (
    select_ada_decode_config,
    use_ada_paired_gqa,
)


def paged_decode_ada(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
    *,
    max_sequence_length: int | None = None,
    length_offset: int = 0,
    block_n: int | None = None,
    num_warps: int | None = None,
    **_,
) -> torch.Tensor:
    """Dispatch to the measured sm_89 tile and K/V-sharing regime."""
    rows, q_heads, _, _ = query.shape
    kv_heads = key_pages.shape[2]
    if max_sequence_length is None:
        # Serving supplies this from host-owned request lengths; this fallback is for
        # direct tests and intentionally synchronizes rather than guessing a regime.
        max_sequence_length = int(seq_lens.max().item()) + length_offset
    if block_n is None or num_warps is None:
        block_n, num_warps = select_ada_decode_config(max_sequence_length, rows)
    gqa_group = q_heads // kv_heads
    # 64x2 is reserved as the paired-GQA regime marker by the Ada selector. This keeps
    # CUDA graph keys distinct across the crossover without a device-side branch.
    paired = (block_n, num_warps) == (64, 2)
    if gqa_group >= 2 and gqa_group % 2 == 0 and paired:
        from engine.kernels.paged_decode_gqa import paged_decode_gqa

        return paged_decode_gqa(
            query, key_pages, value_pages, block_tables, seq_lens,
            scale=scale, block_n=block_n, num_warps=num_warps,
            length_offset=length_offset,
        )

    from engine.kernels.paged_decode_batched import paged_decode_batched

    return paged_decode_batched(
        query, key_pages, value_pages, block_tables, seq_lens,
        scale=scale, block_n=block_n, num_warps=num_warps,
        length_offset=length_offset,
    )
