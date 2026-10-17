"""Search-space W8A16 linear kernel: scale placement, program order and pipelining.

`w8a16_linear.py` dequantizes every `[BLOCK_K, BLOCK_N]` weight tile as `w * scale` in
FP16 inside the reduction loop. Scales are per output channel, so they commute with the
K reduction: `x @ (w * s) == (x @ w) * s`. With `SCALE_AFTER`, INT8 weights convert
exactly to FP16, `tl.dot` accumulates in FP32 and the scale multiplies the accumulator
once, removing a BK x BN multiply per iteration and an FP16 rounding of the product.
`GROUP_M` reorders programs so consecutive programs share weight columns (prefill rows).

This module is an offline-search candidate; `W8A16Linear` keeps its kernel until an
end-to-end A/B decides otherwise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_tuned_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr, M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SCALE_AFTER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * num_n
    group = pid // group_size
    first_m = group * GROUP_M
    rows_in_group = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % rows_in_group
    pid_n = (pid % group_size) // rows_in_group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (start_k + offs_k[None, :]) * stride_xk,
                    mask=(offs_m[:, None] < M) & (start_k + offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + (start_k + offs_k[:, None]) * stride_wk,
                    mask=(offs_n[None, :] < N) & (start_k + offs_k[:, None] < K), other=0).to(tl.float16)
        if SCALE_AFTER:
            acc += tl.dot(x, w)
        else:
            acc += tl.dot(x, w * scales.to(tl.float16)[None, :])
    if SCALE_AFTER:
        acc = acc * scales.to(tl.float32)[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w8a16_linear_tuned(
    inputs: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, *,
    block_m: int, block_n: int, block_k: int, num_warps: int, num_stages: int,
    group_m: int = 1, scale_after: bool = True,
) -> torch.Tensor:
    """`inputs [M, K] @ dequant(qweight [N, K]).T` with an explicit launch configuration."""
    if inputs.ndim != 2 or qweight.ndim != 2 or qweight.dtype is not torch.int8:
        raise ValueError("inputs must be 2D and qweight must be 2D INT8")
    M, K = inputs.shape
    N, weight_k = qweight.shape
    if weight_k != K or scales.shape != (N,):
        raise ValueError("incompatible W8A16 geometry")
    if min(block_m, block_n, block_k) < 16:
        raise ValueError("tl.dot tiles must be at least 16")
    output = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _w8a16_tuned_kernel[grid](
        inputs, qweight, scales, output, M, N, K,
        inputs.stride(0), inputs.stride(1), qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        SCALE_AFTER=scale_after, num_warps=num_warps, num_stages=num_stages,
    )
    return output


# Offline-searched dispatch on RTX 4060 for Qwen3-1.7B MLP shapes (gate/up N=6144 K=2048,
# down N=2048 K=6144), DRAM-streamed timing: results/rtx4060/w8a16_mlp_search_1.7b.json.
# One configuration per row bucket is within 2% of separate gate/up and down configs.
# Above 48 rows a single 32-row tile is used: fused steps run at decode-bucket + 128 x rows
# (129-136, 260-264), where a 64-row tile wasted a third of its last tile and regressed the
# chat fused step 5.6% end to end (w8a16_mlp_search_1.7b_fused_rows.json); 32x64x64 is
# 1.00-1.15x of the current kernel at every measured M from 64 to 264.
_DISPATCH = (
    (16, dict(block_m=16, block_n=32, block_k=128, num_warps=4, num_stages=3, group_m=1, scale_after=False)),
    (48, dict(block_m=32, block_n=128, block_k=128, num_warps=8, num_stages=3, group_m=1, scale_after=False)),
    (None, dict(block_m=32, block_n=64, block_k=64, num_warps=4, num_stages=3, group_m=8, scale_after=True)),
)


def select_w8a16_config(rows: int) -> dict:
    """Deterministic launch configuration for `rows` activation rows (no runtime tuning)."""
    for limit, config in _DISPATCH:
        if limit is None or rows <= limit:
            return config
    raise AssertionError("unreachable")


def w8a16_linear_dispatched(inputs: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                            bias: torch.Tensor | None = None) -> torch.Tensor:
    output = w8a16_linear_tuned(inputs, qweight, scales, **select_w8a16_config(inputs.shape[0]))
    return output if bias is None else output + bias


def install_tuned_w8a16(model, enabled: bool = True) -> int:
    """Switch every `W8A16Linear` in `model` to (or back from) the dispatched kernel.

    The class swap keeps the quantized buffers; only `forward` changes, so engines that
    share one model can A/B the two kernels by toggling before each capture.
    """
    from engine.kernels.w8a16_linear import W8A16Linear

    changed = 0
    for module in model.modules():
        if isinstance(module, W8A16Linear):
            target = TunedW8A16Linear if enabled else W8A16Linear
            if type(module) is not target:
                module.__class__ = target
                changed += 1
    return changed


def _tuned_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    leading = inputs.shape[:-1]
    output = w8a16_linear_dispatched(inputs.reshape(-1, self.in_features), self.qweight,
                                     self.scales, self.bias)
    return output.reshape(*leading, self.out_features)


def _make_tuned_class():
    from engine.kernels.w8a16_linear import W8A16Linear

    return type("TunedW8A16Linear", (W8A16Linear,), {"forward": _tuned_forward,
                                                     "__module__": __name__})


TunedW8A16Linear = _make_tuned_class()


def quantize_lm_head_w8a16(model) -> dict:
    """Give the vocabulary projection its own W8A16 weights (opt-in).

    Under selective MLP W8A16 the FP16 `lm_head` is the largest remaining weight read in a
    decode step: 151936 x 2048 on Qwen3-1.7B, 622 MB, ~2.4 ms. W8A16 halves it (2.41 to
    1.21 ms at 1-16 rows on the RTX 4060). Qwen3 ties `lm_head` to the input embedding, so
    the embedding keeps its FP16 table for lookups and the INT8 copy is extra storage
    (~N x K bytes). Returns both modules so a benchmark can swap them per engine.
    """
    from engine.kernels.w8a16_linear import W8A16Linear

    original = model.lm_head
    if isinstance(original, W8A16Linear):
        return {"fp16": None, "w8a16": original}
    from torch import nn

    from engine.kernels.w8a16_linear import quantize_weight_per_channel

    weight = original.weight.detach()
    # Row chunks: one FP32 pass over a 151936-row vocabulary is a 1.2 GB temporary.
    parts = [quantize_weight_per_channel(weight[start:start + 16384])
             for start in range(0, weight.shape[0], 16384)]
    quantized = TunedW8A16Linear.__new__(TunedW8A16Linear)
    nn.Module.__init__(quantized)
    quantized.in_features, quantized.out_features = original.in_features, original.out_features
    quantized.register_buffer("qweight", torch.cat([q for q, _ in parts]).contiguous())
    quantized.register_buffer("scales", torch.cat([scale for _, scale in parts]).contiguous())
    quantized.register_buffer("bias", None if original.bias is None else original.bias.detach().clone())
    del parts
    model.lm_head = quantized
    return {"fp16": original, "w8a16": quantized}


ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def quantize_attention_w8a16(model, projections: tuple[str, ...] = ATTENTION_PROJECTIONS) -> dict:
    """W8A16 for the attention projections of every layer (opt-in).

    The original rejection used the first kernel, which lost on these narrow shapes. With
    the searched dispatch the four Qwen3-1.7B projections take 70 us against 108 us in
    FP16 per layer at decode width (q/o 1.86x, k/v 1.06-1.2x), and are neutral at a
    132-row fused step. Returns ``{module_path: (fp16_module, w8a16_module)}`` so a
    benchmark can swap arms on one model with `swap_projections`.
    """
    from engine.kernels.w8a16_linear import W8A16Linear
    from engine.model.adapters import attention_modules

    swapped = {}
    found_attention = False
    for path, attention in attention_modules(model):
        found_attention = True
        for name in projections:
            original = getattr(attention, name)
            if isinstance(original, W8A16Linear):
                continue
            quantized = W8A16Linear(original)
            quantized.__class__ = TunedW8A16Linear
            setattr(attention, name, quantized)
            swapped[f"{path}.{name}"] = (original, quantized)
    if not swapped and not found_attention:
        raise RuntimeError("model has no attention projections to quantize")
    return swapped


def swap_projections(model, mapping: dict, *, use_w8a16: bool) -> None:
    """Point each recorded projection at its W8A16 or FP16 module."""
    for path, (fp16, w8a16) in mapping.items():
        parent_path, name = path.rsplit(".", 1)
        setattr(model.get_submodule(parent_path), name, w8a16 if use_w8a16 else fp16)
