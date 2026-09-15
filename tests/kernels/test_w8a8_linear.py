from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
def test_w8a8_linear_matches_per_row_quantization_reference() -> None:
    from engine.kernels.w8a16_linear import quantize_weight_per_channel
    from engine.kernels.w8a8_linear import pack_w8a8_weight, w8a8_linear

    torch.manual_seed(91)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    qweight, wscale = quantize_weight_per_channel(weight)
    scale = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    qx = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    # CUDA does not implement int32 @ int32 in PyTorch. At K=1024 the largest
    # signed INT8 accumulation is 16,516,096 (< 2**24), so FP32 represents this
    # integer reference exactly on the T4.
    reference = (qx.float() @ qweight.t().float()) * scale * wscale.float()[None, :]
    actual = w8a8_linear(x, pack_w8a8_weight(qweight), wscale)
    torch.testing.assert_close(actual, reference.to(torch.float16), rtol=4e-3, atol=4e-3)
