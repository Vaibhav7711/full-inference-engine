"""AWQ GEMM checkpoint repacking for PyTorch's native INT4 matmul."""

from __future__ import annotations

import torch
from torch import nn


def repack_awq_to_tinygemm(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Losslessly convert GEMM-format AWQ nibbles and affine parameters."""
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=qweight.device)
    shifts = 4 * torch.arange(8, device=qweight.device)
    logical = ((qweight[:, :, None] >> shifts) & 15).reshape(qweight.shape[0], -1)
    zeros = ((qzeros[:, :, None] >> shifts) & 15).reshape(qzeros.shape[0], -1)
    channels = torch.arange(logical.shape[1], device=qweight.device).reshape(-1, 8)[:, order].reshape(-1)
    logical = logical[:, channels].T.contiguous()
    zeros = zeros[:, channels]
    packed_nibbles = ((logical[:, 0::2] << 4) | logical[:, 1::2]).to(torch.uint8)
    packed = torch.ops.aten._convert_weight_to_int4pack(packed_nibbles.contiguous(), 8)
    # Native INT4 reconstructs `(q - 8) * scale + floating_zero`.
    floating_zero = (8 - zeros).to(scales.dtype) * scales
    qparams = torch.stack((scales, floating_zero), dim=-1).to(torch.bfloat16).contiguous()
    return packed, qparams


class AWQTinyGemmLinear(nn.Module):
    """Linear layer backed by native packed INT4 weights, with FP16 I/O.

    ``triton_prefill_rows``: when set, inputs with at least this many rows use the
    tensor-core Triton GEMM over the same packed buffers (`engine.kernels.int4pack_gemm`),
    1.8-2.1x faster than tinygemm at 64-132 rows on the RTX 4060; fewer rows (decode)
    keep tinygemm, which wins there. ``dequant_rows``: from this many rows the weight is
    dequantized once into FP16 scratch and multiplied by cuBLAS (4.3x tinygemm at 2048
    rows, the whole-prompt prefill case); None keeps the fused kernel for every prefill.
    Both counts are static per graph shape.
    """

    triton_prefill_rows: int | None = None
    dequant_rows: int | None = None

    def __init__(self, in_features: int, out_features: int, *, group_size: int = 128,
                 device: torch.device | str | None = None):
        super().__init__()
        if in_features % group_size or out_features % 8:
            raise ValueError("native INT4 requires grouped K and eight-channel N alignment")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("packed_weight", torch.empty(
            (out_features // 8, in_features // 128, 32, 4), dtype=torch.int32, device=device,
        ))
        self.register_buffer("scale_and_zero", torch.empty(
            (in_features // group_size, out_features, 2), dtype=torch.bfloat16, device=device,
        ))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features)
        if self.triton_prefill_rows is not None and flat.shape[0] >= self.triton_prefill_rows:
            from engine.kernels.int4pack_gemm import int4pack_gemm_dispatched

            output = int4pack_gemm_dispatched(
                flat, self.packed_weight, self.scale_and_zero,
                fused_min_rows=self.triton_prefill_rows, dequant_min_rows=self.dequant_rows,
            )
            return output.to(inputs.dtype).reshape(*shape, self.out_features)
        output = torch.ops.aten._weight_int4pack_mm(
            flat.to(torch.bfloat16), self.packed_weight, self.group_size, self.scale_and_zero,
        )
        return output.to(inputs.dtype).reshape(*shape, self.out_features)


def install_int4pack_prefill(model: nn.Module, min_rows: int | None = 64,
                            dequant_rows: int | None = 192) -> int:
    """Route prefill-sized AWQ projections off tinygemm (None for `min_rows` disables).

    Thresholds come from `results/rtx4060/int4pack_prefill_search_4b_v3.json`
    (Qwen3-4B-AWQ, all seven projections, DRAM-streamed): the fused kernel from 64 rows,
    dequant+cuBLAS from 192 rows (it wins at 256 on every projection and by 1.8x at 2048).
    """
    changed = 0
    for module in model.modules():
        if isinstance(module, AWQTinyGemmLinear):
            module.triton_prefill_rows = min_rows
            module.dequant_rows = dequant_rows
            changed += 1
    return changed
