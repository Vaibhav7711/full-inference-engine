"""The W4A16 weight format, importable without Triton.

Symmetric 4-bit weights with one fp16 scale per (output channel, group of `group_size`
inputs), packed two per byte. On a card whose memory bandwidth is the decode floor - the
RTX 4060 measures 257 GB/s, the T4 258 - a decode step is a read of every weight once, so
4x fewer weight bytes is the largest single lever this engine has left on either card.

Packing is **split-half within a group**, not interleaved: for a group of G inputs, byte
`j` (0 <= j < G/2) holds the quantised weight for input `j` in its low nibble and for
input `j + G/2` in its high nibble. A kernel that loads one `[G/2, BLOCK_N]` byte tile
therefore unpacks into two contiguous `[G/2, BLOCK_N]` weight tiles with no shuffle, and
the K reduction over the group is two `tl.dot`s - one against the first half of the
activation tile, one against the second. Interleaved packing would need a register
permutation Triton cannot express cheaply.

Values are stored offset by 8 (`q + 8` in 0..15) and clamped to the symmetric range
[-7, 7], the same convention as the INT8 path's [-127, 127].
"""

from __future__ import annotations

import torch

GROUP_SIZE = 128
Q_MAX = 7.0
OFFSET = 8


def quantize_weight_w4_grouped(
    weight: torch.Tensor, group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[N, K]` fp16/fp32 weights -> `(packed uint8 [N, K//2], scales fp16 [N, K//group])`."""
    if weight.ndim != 2:
        raise ValueError("weight must be [out_features, in_features]")
    out_features, in_features = weight.shape
    if group_size % 2 or in_features % group_size:
        raise ValueError(f"in_features {in_features} must be a multiple of an even group_size "
                         f"({group_size})")
    groups = in_features // group_size
    half = group_size // 2
    values = weight.float().reshape(out_features, groups, group_size)
    scales = values.abs().amax(dim=-1).clamp_min(1e-8) / Q_MAX              # [N, groups]
    q = torch.round(values / scales[..., None]).clamp(-Q_MAX, Q_MAX) + OFFSET  # 1..15
    q = q.to(torch.uint8)
    low, high = q[..., :half], q[..., half:]                                 # [N, groups, half]
    packed = (low | (high << 4)).reshape(out_features, groups * half)       # [N, K//2]
    return packed.contiguous(), scales.to(torch.float16).contiguous()


def unpack_w4_reference(
    packed: torch.Tensor, scales: torch.Tensor, in_features: int, group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Dequantise `(packed, scales)` back to fp16 `[N, K]` - the kernel's contract."""
    if packed.dtype is not torch.uint8:
        raise ValueError("packed weights must be uint8")
    out_features = packed.shape[0]
    groups = in_features // group_size
    half = group_size // 2
    bytes_ = packed.reshape(out_features, groups, half)
    low = (bytes_ & 0x0F).to(torch.int16) - OFFSET
    high = (bytes_ >> 4).to(torch.int16) - OFFSET
    q = torch.cat((low, high), dim=-1).float()                               # [N, groups, G]
    return (q * scales.float()[..., None]).reshape(out_features, in_features).to(torch.float16)


def weight_bytes(in_features: int, out_features: int, group_size: int = GROUP_SIZE) -> dict[str, int]:
    """Bytes a decode step reads for one projection, by format - the point of all this."""
    fp16 = 2 * in_features * out_features
    int8 = in_features * out_features + 2 * out_features
    w4 = in_features * out_features // 2 + 2 * out_features * (in_features // group_size)
    return {"fp16": fp16, "w8a16": int8, "w4a16": w4}
