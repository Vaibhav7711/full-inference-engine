from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
def test_w8a8_linear_matches_tiled_quantization_reference() -> None:
    from engine.kernels.w8a16_linear import quantize_weight_per_channel
    from engine.kernels.w8a8_linear import w8a8_linear

    torch.manual_seed(91)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    qweight, wscale = quantize_weight_per_channel(weight)
    reference = torch.zeros(16, 1024, device="cuda", dtype=torch.float32)
    for start in range(0, 1024, 32):
        tile = x[:, start:start + 32].float()
        scale = tile.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        qx = torch.round(tile / scale).clamp(-127, 127)
        reference += (qx @ qweight[:, start:start + 32].t().float()) * scale * wscale.float()[None, :]
    actual = w8a8_linear(x, qweight, wscale)
    torch.testing.assert_close(actual, reference.to(torch.float16), rtol=4e-3, atol=4e-3)
