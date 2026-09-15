"""Fused SiLU-gated linear unit activation and model-module installer."""

from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
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
    gate = module.gate_proj(hidden_states)
    up = module.up_proj(hidden_states)
    return module.down_proj(triton_swiglu(gate, up))


def install_triton_qwen_swiglu(model: torch.nn.Module) -> int:
    """Patch Qwen3 MLP modules and return the number installed."""
    installed = 0
    for module in model.modules():
        if module.__class__.__name__.lower() != "qwen3mlp":
            continue
        if not hasattr(module, "_pre_triton_swiglu_forward"):
            module._pre_triton_swiglu_forward = module.forward
            module.forward = MethodType(_triton_qwen_mlp_forward, module)
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
        module.forward = original
        del module._pre_triton_swiglu_forward
        restored += 1
    return restored
