"""Fused SiLU-gated linear unit activation and model-module installer."""

from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl
from torch import nn
from torch.nn import functional as F


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    # Triton 3.6's sigmoid accepts FP32/FP64. Promotion also mirrors the stable
    # internal evaluation used by PyTorch SiLU before the result is stored as FP16.
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, gate * tl.sigmoid(gate) * up, mask=mask)


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute `silu(gate) * up` in one elementwise kernel."""
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("gate and up tensors must have matching shape, dtype, and device")
    if gate.device.type != "cuda":
        raise ValueError("fused SwiGLU requires CUDA tensors")
    gate = gate.contiguous()
    up = up.contiguous()
    output = torch.empty_like(gate)
    elements = gate.numel()
    _swiglu_kernel[(triton.cdiv(elements, 256),)](
        gate, up, output, elements, BLOCK=256, num_warps=4
    )
    return output


def _triton_qwen_mlp_forward(module, hidden_states: torch.Tensor) -> torch.Tensor:
    fused_projection = getattr(module, "fused_gate_up_proj", None)
    if fused_projection is None:
        gate = module.gate_proj(hidden_states)
        up = module.up_proj(hidden_states)
    else:
        gate, up = fused_projection(hidden_states)
    return module.down_proj(triton_swiglu(gate, up))


class FusedGateUpProjection(nn.Module):
    """One linear projection that returns the gate and up halves for Qwen SwiGLU."""

    def __init__(self, gate: nn.Linear, up: nn.Linear):
        super().__init__()
        if (gate.in_features != up.in_features or gate.out_features != up.out_features
                or (gate.bias is None) != (up.bias is None)):
            raise ValueError("gate/up projections must have matching geometry and bias layout")
        self.gate_features = gate.out_features
        self.register_buffer("weight", torch.cat((gate.weight.detach(), up.weight.detach()), dim=0))
        if gate.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", torch.cat((gate.bias.detach(), up.bias.detach()), dim=0))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = F.linear(hidden_states, self.weight, self.bias)
        return output.split(self.gate_features, dim=-1)


def install_triton_qwen_swiglu(model: torch.nn.Module, *, fuse_gate_up: bool = True) -> int:
    """Patch Qwen3 MLP modules and return the number installed."""
    installed = 0
    for module in model.modules():
        if module.__class__.__name__.lower() != "qwen3mlp":
            continue
        if not hasattr(module, "_pre_triton_swiglu_forward"):
            module._pre_triton_swiglu_forward = module.forward.__func__
            module.forward = MethodType(_triton_qwen_mlp_forward, module)
            if fuse_gate_up:
                gate, up = module.gate_proj, module.up_proj
                fused = FusedGateUpProjection(gate, up)
                # Keep originals outside Module registration so fusion actually removes
                # their duplicate parameters from the active inference model, while still
                # allowing an exact uninstall for tests/debugging.
                module.__dict__["_pre_triton_gate_proj"] = gate
                module.__dict__["_pre_triton_up_proj"] = up
                module.gate_proj = None
                module.up_proj = None
                module.fused_gate_up_proj = fused
        installed += 1
    if installed == 0:
        raise RuntimeError("model contains no Qwen3MLP modules")
    return installed


def uninstall_triton_qwen_swiglu(model: torch.nn.Module) -> int:
    restored = 0
    for module in model.modules():
        original = getattr(module, "_pre_triton_swiglu_forward", None)
        if original is None:
            continue
        module.forward = MethodType(original, module)
        del module._pre_triton_swiglu_forward
        original_gate = module.__dict__.pop("_pre_triton_gate_proj", None)
        original_up = module.__dict__.pop("_pre_triton_up_proj", None)
        if original_gate is not None and original_up is not None:
            module.gate_proj = original_gate
            module.up_proj = original_up
            module.fused_gate_up_proj = None
        restored += 1
    return restored
