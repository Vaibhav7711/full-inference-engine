"""Device-independent policy for selecting paged decode kernel regimes."""

from __future__ import annotations


DECODE_ATTENTION_KINDS = ("per_head", "gqa")


def select_paged_decode_config(
    max_sequence_length: int, batch_size: int, kernel: str = "per_head",
) -> tuple[int, int]:
    """Select the measured T4 decode tile/warp regime.

    The T4 sweep showed 64x4 wins at a 64-token context, while 128x4 improves
    kernel latency by 20-42% at 256-2048 tokens across widths 1, 8, and 16.
    Mixed-length batches select from their longest active sequence so every program
    uses one valid compile-time configuration without a device-side synchronization.

    The GQA-shared kernel holds two heads of state over the same rank-2 tiles, so it
    takes the per-head regime as its starting point. Re-derive from
    `paged_decode_regime_sweep.py --kernel both` on a new device.
    """
    if max_sequence_length <= 0 or batch_size <= 0:
        raise ValueError("max_sequence_length and batch_size must be positive")
    if kernel not in DECODE_ATTENTION_KINDS:
        raise ValueError(f"kernel must be one of {DECODE_ATTENTION_KINDS}")
    block_n = 128 if max_sequence_length >= 128 else 64
    return block_n, 4
