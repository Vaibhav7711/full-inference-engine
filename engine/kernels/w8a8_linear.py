"""Experimental true W8A8 decode linear for Turing through CUDA INT8 GEMM.

Triton 3.x cannot lower signed INT8 MMA for sm75 reliably. ``torch._int_mm``
dispatches to CUDA's native INT8 GEMM implementation instead, so this remains
a genuine INT8-times-INT8 accumulation experiment on a T4.
"""

from __future__ import annotations

import torch
def w8a8_linear(inputs: torch.Tensor, qweight: torch.Tensor, weight_scales: torch.Tensor) -> torch.Tensor:
    """Dynamically quantize each input row, execute CUDA INT8 GEMM, dequantize."""
    if inputs.ndim != 2 or qweight.dtype is not torch.int8 or qweight.ndim != 2:
        raise ValueError("inputs must be 2D and qweight must be 2D INT8")
    _, width = inputs.shape
    output_width, weight_width = qweight.shape
    if weight_width != width or weight_scales.shape != (output_width,):
        raise ValueError("invalid W8A8 geometry")
    if inputs.device.type != "cuda" or qweight.device != inputs.device or weight_scales.device != inputs.device:
        raise ValueError("all W8A8 tensors must be on the same CUDA device")
    activation_scales = inputs.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    qinputs = torch.round(inputs.float() / activation_scales).clamp(-127, 127).to(torch.int8).contiguous()
    # torch._int_mm is CUDA's INT8 x INT8 -> INT32 GEMM. qweight is output-major
    # for a regular linear layer, hence the transpose for GEMM.
    accumulators = torch._int_mm(qinputs, qweight.t().contiguous())
    return (accumulators.float() * activation_scales * weight_scales.float()[None, :]).to(inputs.dtype)
