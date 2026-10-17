"""Fused SiLU-gated linear unit activation and model-module installer."""

from __future__ import annotations

import sys
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


def _triton_mlp_forward(module, hidden_states: torch.Tensor) -> torch.Tensor:
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


class FusedQKVProjection(nn.Module):
    """One packed Q/K/V projection for Qwen/Llama-style attention modules.

    This is deliberately an FP16 execution experiment: it removes two GEMM launches
    and presents cuBLASLt with one wider output matrix, but it does not reduce weight
    bytes.  It therefore remains opt-in until the end-to-end A/B says otherwise.
    """

    def __init__(self, q: nn.Linear, k: nn.Linear, v: nn.Linear):
        super().__init__()
        if not (q.in_features == k.in_features == v.in_features):
            raise ValueError("Q/K/V projections must have a common input width")
        if len({q.bias is None, k.bias is None, v.bias is None}) != 1:
            raise ValueError("Q/K/V projections must have matching bias layouts")
        self.q_features, self.k_features, self.v_features = (
            q.out_features, k.out_features, v.out_features,
        )
        self.register_buffer("weight", torch.cat((q.weight.detach(), k.weight.detach(), v.weight.detach()), dim=0))
        if q.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", torch.cat((q.bias.detach(), k.bias.detach(), v.bias.detach()), dim=0))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = F.linear(hidden_states, self.weight, self.bias)
        return output.split((self.q_features, self.k_features, self.v_features), dim=-1)


def _fused_qkv_attention_forward(
    module, hidden_states: torch.Tensor, position_embeddings, attention_mask,
    past_key_values=None, **kwargs,
):
    """The stock Qwen3 attention body with only its three projections replaced."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query, key, value = module.fused_qkv_proj(hidden_states)
    query_states = module.q_norm(query.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(key.view(hidden_shape)).transpose(1, 2)
    value_states = value.view(hidden_shape).transpose(1, 2)

    rope_module = sys.modules[type(module).__module__]
    query_states, key_states = rope_module.apply_rotary_pos_emb(
        query_states, key_states, *position_embeddings,
    )
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, rope_module.eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        module, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling, sliding_window=module.sliding_window, **kwargs,
    )
    return module.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), attn_weights


def install_fused_qkv(model: torch.nn.Module) -> int:
    """Opt in to fused QKV projections for Qwen3 attention modules.

    The forward contract is intentionally checked structurally. Other families keep
    their stock attention forward rather than accepting an almost-compatible patch.
    """
    installed = 0
    for module in model.modules():
        if not all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj")):
            continue
        if hasattr(module, "_pre_fused_qkv_forward"):
            installed += 1
            continue
        q, k, v = module.q_proj, module.k_proj, module.v_proj
        if not all(isinstance(proj, nn.Linear) for proj in (q, k, v)):
            continue
        module._pre_fused_qkv_forward = module.forward.__func__
        module.__dict__["_pre_fused_q_proj"] = q
        module.__dict__["_pre_fused_k_proj"] = k
        module.__dict__["_pre_fused_v_proj"] = v
        module.q_proj = None
        module.k_proj = None
        module.v_proj = None
        module.fused_qkv_proj = FusedQKVProjection(q, k, v)
        module.forward = MethodType(_fused_qkv_attention_forward, module)
        installed += 1
    if not installed:
        raise RuntimeError("model contains no compatible Q/K/V attention projections")
    return installed


def uninstall_fused_qkv(model: torch.nn.Module) -> int:
    restored = 0
    for module in model.modules():
        original = getattr(module, "_pre_fused_qkv_forward", None)
        if original is None:
            continue
        module.forward = MethodType(original, module)
        del module._pre_fused_qkv_forward
        module.q_proj = module.__dict__.pop("_pre_fused_q_proj")
        module.k_proj = module.__dict__.pop("_pre_fused_k_proj")
        module.v_proj = module.__dict__.pop("_pre_fused_v_proj")
        module.fused_qkv_proj = None
        restored += 1
    return restored


def install_triton_swiglu(model: torch.nn.Module, *, fuse_gate_up: bool = False) -> int:
    """Patch every SwiGLU feed-forward in the model and return the number installed.

    Modules are found structurally - anything with `gate_proj`, `up_proj` and
    `down_proj` - so this covers Qwen, Llama, Mistral and every other Llama-style MLP
    without a per-family name list. See `engine/model/adapters.py`.
    """
    from engine.model.adapters import mlp_modules

    installed = 0
    for module in mlp_modules(model):
        if not hasattr(module, "_pre_triton_swiglu_forward"):
            module._pre_triton_swiglu_forward = module.forward.__func__
            module.forward = MethodType(_triton_mlp_forward, module)
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
        raise RuntimeError(
            "model contains no gate/up/down SwiGLU modules; this engine's MLP fusion "
            "does not apply to its feed-forward"
        )
    return installed


def uninstall_triton_swiglu(model: torch.nn.Module) -> int:
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


# The Qwen-specific names these installers used to have. Kept so older benchmarks and
# notebooks keep working; they are the same functions.
install_triton_qwen_swiglu = install_triton_swiglu
uninstall_triton_qwen_swiglu = uninstall_triton_swiglu
