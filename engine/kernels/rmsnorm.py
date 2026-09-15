"""Fused Triton RMSNorm for inference-time model normalization."""

from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    epsilon,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    values = tl.load(
        input_ptr + row * input_row_stride + columns, mask=mask, other=0.0
    ).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / WIDTH
    inverse_rms = tl.rsqrt(variance + epsilon)
    # Match the Qwen reference ordering: normalize in FP32, cast back to the input
    # dtype, then apply the model weight in that dtype.
    normalized = (values * inverse_rms).to(input_ptr.dtype.element_ty)
    weights = tl.load(weight_ptr + columns, mask=mask, other=0.0)
    normalized = normalized * weights
    tl.store(
        output_ptr + row * output_row_stride + columns, normalized, mask=mask
    )


def triton_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Apply RMSNorm over the final dimension with FP32 reduction and output dtype."""
    if hidden_states.ndim < 1 or hidden_states.shape[-1] == 0:
        raise ValueError("hidden states must have a non-empty final dimension")
    width = hidden_states.shape[-1]
    if weight.ndim != 1 or weight.numel() != width:
        raise ValueError("RMSNorm weight must match the final input dimension")
    if hidden_states.device.type != "cuda" or weight.device != hidden_states.device:
        raise ValueError("RMSNorm input and weight must share a CUDA device")
    if weight.dtype != hidden_states.dtype:
        raise ValueError("RMSNorm input and weight must have the same dtype")
    if width > 65536:
        raise ValueError("RMSNorm width exceeds the supported Triton block size")

    source = hidden_states.contiguous()
    weights = weight.contiguous()
    output = torch.empty_like(source)
    rows = source.numel() // width
    block = triton.next_power_of_2(width)
    warps = 8 if block >= 2048 else 4
    _rmsnorm_kernel[(rows,)](
        source,
        weights,
        output,
        source.stride(-2) if source.ndim > 1 else width,
        output.stride(-2) if output.ndim > 1 else width,
        epsilon,
        WIDTH=width,
        BLOCK=block,
        num_warps=warps,
    )
    return output.view(hidden_states.shape)


def _triton_rmsnorm_forward(module, hidden_states: torch.Tensor) -> torch.Tensor:
    epsilon = getattr(module, "variance_epsilon", getattr(module, "eps", None))
    if epsilon is None:
        raise RuntimeError("patched RMSNorm module has no epsilon attribute")
    return triton_rmsnorm(hidden_states, module.weight, float(epsilon))


def install_triton_rmsnorm(model: torch.nn.Module) -> int:
    """Patch recognized RMSNorm modules in-place and return the installed count."""
    installed = 0
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        is_rmsnorm = class_name.endswith("rmsnorm")
        has_contract = (
            isinstance(getattr(module, "weight", None), torch.Tensor)
            and (
                hasattr(module, "variance_epsilon")
                or hasattr(module, "eps")
            )
        )
        if not is_rmsnorm or not has_contract:
            continue
        if hasattr(module, "_pre_triton_rmsnorm_forward"):
            installed += 1
            continue
        # Store the unbound function, not a bound method that references `module`
        # from an attribute on itself and creates a GPU-model retention cycle.
        module._pre_triton_rmsnorm_forward = module.forward.__func__
        module.forward = MethodType(_triton_rmsnorm_forward, module)
        installed += 1
    if installed == 0:
        raise RuntimeError("model contains no recognized RMSNorm modules")
    return installed


def uninstall_triton_rmsnorm(model: torch.nn.Module) -> int:
    """Restore RMSNorm forward methods previously replaced by the installer."""
    restored = 0
    for module in model.modules():
        original = getattr(module, "_pre_triton_rmsnorm_forward", None)
        if original is None:
            continue
        module.forward = MethodType(original, module)
        del module._pre_triton_rmsnorm_forward
        restored += 1
    return restored
