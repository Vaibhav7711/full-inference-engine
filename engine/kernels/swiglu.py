"""Fused SiLU-gated linear unit activation and model-module installer."""

from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl
from torch import nn
from torch.nn import functional as F


@triton.jit
def _swiglu_kernel(
    gate_ptr, up_ptr, out_ptr, columns,
    stride_gate, stride_up, stride_out,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < columns
    # Triton 3.6's sigmoid accepts FP32/FP64. Promotion also mirrors the stable
    # internal evaluation used by PyTorch SiLU before the result is stored as FP16.
    gate = tl.load(gate_ptr + row * stride_gate + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + row * stride_up + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_out + offsets, gate * tl.sigmoid(gate) * up, mask=mask)


def _as_rows(tensor: torch.Tensor) -> torch.Tensor:
    """View `[..., columns]` as `[rows, columns]` without copying when the layout allows.

    The fused gate/up projection hands over the two halves of one `[..., 2*columns]`
    output as strided views. Reading them in place through a row stride is what makes
    that fusion save a launch instead of paying two full activation copies to make the
    halves contiguous first.
    """
    if tensor.stride(-1) != 1:
        tensor = tensor.contiguous()
    return tensor.reshape(-1, tensor.shape[-1])


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute `silu(gate) * up` in one elementwise kernel."""
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("gate and up tensors must have matching shape, dtype, and device")
    if gate.device.type != "cuda":
        raise ValueError("fused SwiGLU requires CUDA tensors")
    if gate.ndim == 0 or gate.numel() == 0:
        return torch.nn.functional.silu(gate) * up
    gate_rows = _as_rows(gate)
    up_rows = _as_rows(up)
    rows, columns = gate_rows.shape
    output = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    output_rows = output.view(rows, columns)
    block = 512
    _swiglu_kernel[(rows, triton.cdiv(columns, block))](
        gate_rows, up_rows, output_rows, columns,
        gate_rows.stride(0), up_rows.stride(0), output_rows.stride(0),
        BLOCK=block, num_warps=4,
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


def install_triton_qwen_swiglu(model: torch.nn.Module, *, fuse_gate_up: bool = False) -> int:
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
