"""Experimental W8A16 linear kernel for decode-sized Qwen projections."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _w8a16_linear_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_on, stride_om,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # One output-channel scale applies to every K tile; keep it in registers rather
    # than reloading it inside the reduction loop.
    scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float16)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (start_k + offs_k[None, :]) * stride_xk,
            mask=(offs_m[:, None] < M) & (start_k + offs_k[None, :] < K), other=0.0,
        )
        weight = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + (start_k + offs_k[:, None]) * stride_wk,
            mask=(offs_n[None, :] < N) & (start_k + offs_k[:, None] < K), other=0,
        ).to(tl.float16)
        accumulator += tl.dot(x, weight * scales[None, :])
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             accumulator.to(out_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack `[out_features, in_features]` FP16 weights into INT8 plus FP16 scales."""
    scales = weight.float().abs().amax(dim=1).clamp_min(1e-8).div(127).to(torch.float16)
    quantized = torch.round(weight.float() / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scales.contiguous()


def w8a16_linear(inputs: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                 bias: torch.Tensor | None = None, *, block_m: int = 16,
                 block_n: int = 128, block_k: int = 64,
                 num_warps: int = 4) -> torch.Tensor:
    """W8A16 linear for 2D decode activations; dequantization happens in the kernel."""
    if inputs.ndim != 2 or qweight.ndim != 2 or qweight.dtype is not torch.int8:
        raise ValueError("inputs must be 2D and qweight must be 2D INT8")
    M, K = inputs.shape
    N, weight_k = qweight.shape
    if weight_k != K or scales.shape != (N,) or (bias is not None and bias.shape != (N,)):
        raise ValueError("incompatible W8A16 linear geometry")
    if K % 32:
        raise ValueError("W8A16 kernel requires input width divisible by 32")
    output = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)
    if block_m not in {8, 16, 32} or block_n not in {32, 64, 128, 256} \
            or block_k not in {32, 64, 128} or num_warps not in {2, 4, 8}:
        raise ValueError("unsupported W8A16 launch configuration")
    _w8a16_linear_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](
        inputs, qweight, scales, bias if bias is not None else output, output,
        M, N, K, inputs.stride(0), inputs.stride(1), qweight.stride(0), qweight.stride(1),
        output.stride(1), output.stride(0), HAS_BIAS=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps,
    )
    return output


class W8A16Linear(nn.Module):
    """Drop-in linear using persistent per-output-channel INT8 weights.

    The Triton kernel consumes two-dimensional rows, while transformer projections
    receive both ``[batch, hidden]`` and ``[batch, sequence, hidden]`` tensors.  Keep
    that adaptation here so model-level experiments exercise exactly the same kernel
    as the projection microbenchmark.
    """

    def __init__(self, source: nn.Linear):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        qweight, scales = quantize_weight_per_channel(source.weight.detach())
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {inputs.shape[-1]}"
            )
        leading = inputs.shape[:-1]
        output = w8a16_linear(
            inputs.reshape(-1, self.in_features), self.qweight, self.scales, self.bias,
        )
        return output.reshape(*leading, self.out_features)


def quantize_mlp_w8a16(model: nn.Module, *, tuned: bool = True) -> int:
    """Replace only gate/up/down projections in dense SwiGLU blocks.

    Attention projections are intentionally excluded: the RTX 4060 projection gate
    found W8A16 slower for Q/K/V at every tested decode batch. ``tuned`` selects the
    offline-searched dispatch in `engine.kernels.w8a16_tuned` (Qwen3-1.7B, RTX 4060:
    decode -3.5% at batch 1, -4.6% at concurrency 6, end to end); False keeps the
    original single-configuration kernel.
    """
    from engine.model.adapters import mlp_modules

    replaced = 0
    for mlp in mlp_modules(model):
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(mlp, name)
            if isinstance(projection, W8A16Linear):
                continue
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"{type(mlp).__name__}.{name} is not nn.Linear")
            setattr(mlp, name, W8A16Linear(projection))
            replaced += 1
    if replaced == 0:
        raise RuntimeError("model contains no dense SwiGLU MLP projections to quantize")
    model.config._weight_only_int8_backend = "triton_w8a16_mlp"
    from engine.kernels.w8a16_tuned import install_tuned_w8a16
    install_tuned_w8a16(model, enabled=tuned)
    return replaced
