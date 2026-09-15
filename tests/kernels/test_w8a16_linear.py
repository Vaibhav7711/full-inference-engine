"""Correctness gate for the experimental fused-scale W8A16 linear kernel."""

from __future__ import annotations

import pytest
import torch


cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
@pytest.mark.parametrize("shape", [(16, 896, 896), (16, 896, 4864), (1, 896, 896)])
def test_w8a16_linear_matches_dequantized_reference(shape: tuple[int, int, int]) -> None:
    from engine.kernels.w8a16_linear import quantize_weight_per_channel, w8a16_linear

    torch.manual_seed(sum(shape))
    rows, width, outputs = shape
    inputs = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    weight = torch.randn(outputs, width, device="cuda", dtype=torch.float16)
    bias = torch.randn(outputs, device="cuda", dtype=torch.float16)
    qweight, scales = quantize_weight_per_channel(weight)
    reference = torch.nn.functional.linear(inputs, qweight.to(torch.float16) * scales[:, None], bias)
    actual = w8a16_linear(inputs, qweight, scales, bias)
    torch.testing.assert_close(actual, reference, rtol=3e-3, atol=3e-3)
