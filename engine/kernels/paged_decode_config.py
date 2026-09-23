"""Device-independent policy for selecting paged decode kernel regimes."""

from __future__ import annotations

# Kinds this policy has a measured regime for. Other backends (flash, which tiles
# internally) accept the values and ignore them.
DECODE_ATTENTION_KINDS = ("per_head", "gqa", "split_k")


def select_paged_decode_config(
    max_sequence_length: int, batch_size: int, kernel: str = "per_head",
) -> tuple[int, int]:
    """Select the measured T4 decode tile/warp regime.

    The T4 sweep showed 64x4 wins at a 64-token context, while 128x4 improves
    kernel latency by 20-42% at 256-2048 tokens across widths 1, 8, and 16.
    Mixed-length batches select from their longest active sequence so every program
    uses one valid compile-time configuration without a device-side synchronization.

    The GQA-shared and split-K kernels hold the same rank-2 tiles, so they start from the
    per-head regime; a kernel that tiles internally (flash) ignores these values. Nothing
    here is device-specific beyond the T4 sweep that produced it - re-derive with
    `paged_decode_regime_sweep.py` on a new architecture.
    """
    if max_sequence_length <= 0 or batch_size <= 0:
        raise ValueError("max_sequence_length and batch_size must be positive")
    block_n = 128 if max_sequence_length >= 128 else 64
    return block_n, 4


def choose_splits(
    rows: int, q_heads: int, max_sequence_length: int, *,
    multiprocessors: int = 40, block_n: int = 64, limit: int = 16,
) -> int:
    """How many slices to cut each row's keys into, or 1 to use the single-pass kernel.

    Two bounds. The grid must be wide enough to fill the device - `rows * heads * splits`
    at least two blocks per SM - and each slice must be worth a launch, so it gets at
    least one full `BLOCK_N` tile. Powers of two only, because the merge kernel's
    `tl.arange(0, SPLITS)` needs one.
    """
    if rows <= 0 or q_heads <= 0 or max_sequence_length <= 0:
        raise ValueError("rows, heads and sequence length must be positive")
    programs = rows * q_heads
    wanted = max(1, -(-2 * multiprocessors // programs))
    by_length = max(1, max_sequence_length // block_n)
    splits = min(wanted, by_length, limit)
    return 1 << (splits.bit_length() - 1) if splits > 0 else 1
