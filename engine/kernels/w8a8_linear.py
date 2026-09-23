"""Experimental true W8A8 decode linear for Turing through CUDA INT8 GEMM.

Triton 3.x cannot lower signed INT8 MMA for sm75 reliably. ``torch._int_mm``
dispatches to CUDA's native INT8 GEMM implementation instead, so this remains
a genuine INT8-times-INT8 accumulation experiment on a T4.
"""

from __future__ import annotations

import torch


def pack_w8a8_weight(qweight: torch.Tensor) -> torch.Tensor:
    """Prepack [out_features, in_features] INT8 weights for CUDA GEMM once."""
    if qweight.ndim != 2 or qweight.dtype is not torch.int8:
        raise ValueError("qweight must be a 2D INT8 tensor")
    return qweight.t().contiguous()


def _int8_gemm_with_minimum_rows(qinputs: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    """Run CUDA INT8 GEMM on CUDA backends that require an aligned M dimension.

    ``torch._int_mm`` documents M > 16, but the cuBLASLt path used by CUDA 13 on Ada
    also rejects the minimally valid 17-row launch.  Padding to one 32-row tile works on
    both that path and the T4 path this kernel was originally written for.
    """
    minimum_rows = 32
    if qinputs.shape[0] >= minimum_rows:
        return torch._int_mm(qinputs, packed_weight)
    padded = torch.zeros((minimum_rows, qinputs.shape[1]), device=qinputs.device, dtype=torch.int8)
    padded[:qinputs.shape[0]].copy_(qinputs)
    return torch._int_mm(padded, packed_weight)[:qinputs.shape[0]]


def w8a8_linear(inputs: torch.Tensor, packed_weight: torch.Tensor, weight_scales: torch.Tensor) -> torch.Tensor:
    """Dynamically quantize each input row, execute CUDA INT8 GEMM, dequantize."""
    if inputs.ndim != 2 or packed_weight.dtype is not torch.int8 or packed_weight.ndim != 2:
        raise ValueError("inputs must be 2D and packed_weight must be 2D INT8")
    _, width = inputs.shape
    weight_width, output_width = packed_weight.shape
    if weight_width != width or weight_scales.shape != (output_width,):
        raise ValueError("invalid W8A8 geometry")
    if inputs.device.type != "cuda" or packed_weight.device != inputs.device or weight_scales.device != inputs.device:
        raise ValueError("all W8A8 tensors must be on the same CUDA device")
    activation_scales = inputs.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    qinputs = torch.round(inputs.float() / activation_scales).clamp(-127, 127).to(torch.int8).contiguous()
    # torch._int_mm is CUDA's INT8 x INT8 -> INT32 GEMM. packed_weight is
    # deliberately pretransposed, avoiding a per-token weight copy.
    accumulators = _int8_gemm_with_minimum_rows(qinputs, packed_weight)
    return (accumulators.float() * activation_scales * weight_scales.float()[None, :]).to(inputs.dtype)
