"""Reference per-output-channel INT8 weight-only quantization.

This is an accuracy and memory experiment, not a claim of an optimized INT8 GEMM.
`Int8Linear` dequantizes to the activation dtype for `F.linear`; a later kernel-backed
implementation is required before expecting throughput gains.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Int8Linear(nn.Module):
    def __init__(self, source: nn.Linear):
        super().__init__()
        weight = source.weight.detach()
        scales = weight.float().abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps) / 127.0
        qweight = torch.round(weight.float() / scales[:, None]).clamp(-127, 127).to(torch.int8)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(dtype=inputs.dtype) * self.scales.to(dtype=inputs.dtype)[:, None]
        bias = self.bias.to(dtype=inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, weight, bias)


def quantize_linear_modules(module: nn.Module) -> int:
    """Replace every `nn.Linear` child recursively; returns the replacement count."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child))
            replaced += 1
        else:
            replaced += quantize_linear_modules(child)
    return replaced


def model_storage_bytes(module: nn.Module) -> int:
    """Sum unique parameter and buffer storages, avoiding tied-weight double counting."""
    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.device.type == "meta":
            continue
        key = (str(tensor.device), tensor.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total += tensor.numel() * tensor.element_size()
    return total
