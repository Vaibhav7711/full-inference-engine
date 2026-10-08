"""The FP8 (E4M3) storage contract, importable without Triton.

Everything a CPU test or the engine's constructor needs to reason about FP8 KV lives
here: the dtype, the largest finite value, where the format can run, and a pure-torch
reference quantiser the kernels in :mod:`fp8_paged_kv` are checked against. Keeping it
Triton-free is what lets the arithmetic be pinned on a machine with no GPU before the
kernels are ever launched.

Scale convention: `scale = max(|x|) / 448` with `448` the largest finite E4M3 value, and
the quantised value is clamped to +-448 before the cast. E4M3 has no infinity - an
overflowing cast produces NaN - so the clamp is correctness, not tidiness.
"""

from __future__ import annotations

import torch

# Largest finite value representable in float8_e4m3fn.
E4M3_MAX = 448.0
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def unavailable_reason(profile) -> str | None:
    """Why FP8 KV cannot run on this device, or None when it can.

    The `float8e4nv` type exists in Triton for sm_89 and later. Ada has FP8 tensor cores,
    but here FP8 is a storage format only - the dequantising load is what gates it.
    """
    if FP8_DTYPE is None:
        return "this torch build has no float8_e4m3fn dtype"
    if profile is not None and profile.sm < 89:
        return f"FP8 (E4M3) storage needs sm_89+; this device is sm_{profile.sm}"
    return None


def quantize_fp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector E4M3 quantisation over the last dim: `(quantised, scale)`.

    `scale` has `x.shape[:-1]`; dequantise with `q.float() * scale[..., None]`.
    """
    if FP8_DTYPE is None:
        raise RuntimeError("this torch build has no float8_e4m3fn dtype")
    values = x.float()
    # Same order as the kernels: divide, then floor the scale. An all-zero vector gets
    # scale 1e-8 and quantises to zeros in both.
    scale = (values.abs().amax(dim=-1) / E4M3_MAX).clamp_min(1e-8)
    quantised = (values / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE)
    return quantised, scale


def dequantize_fp8_reference(quantised: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantised.float() * scale.float()[..., None]
