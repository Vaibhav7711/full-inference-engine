"""The Triton int4pack GEMM reads tinygemm's packed layout exactly."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _packed(n: int, k: int):
    q = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    nibbles = ((q[:, 0::2] << 4) | q[:, 1::2]).to(torch.uint8).contiguous()
    packed = torch.ops.aten._convert_weight_to_int4pack(nibbles, 8)
    scale = torch.rand(k // 128, n, device="cuda") * 0.02 + 0.001
    zero = (torch.rand(k // 128, n, device="cuda") - 0.5) * 0.1
    qparams = torch.stack((scale, zero), -1).to(torch.bfloat16).contiguous()
    dense = (q.float() - 8) * qparams[..., 0].float().T.repeat_interleave(128, 1) \
        + qparams[..., 1].float().T.repeat_interleave(128, 1)
    return packed, qparams, dense


@cuda
@requires_cuda
@pytest.mark.parametrize("rows", [1, 64, 133])
@pytest.mark.parametrize("shape", [(1024, 2560), (2560, 9728), (136, 256)])
def test_int4pack_gemm_matches_exact_dequantization_and_tinygemm(rows, shape) -> None:
    from engine.kernels.int4pack_gemm import int4pack_gemm, int4pack_gemm_v2

    torch.manual_seed(rows + shape[0])
    packed, qparams, dense = _packed(*shape)
    x = torch.randn(rows, shape[1], device="cuda", dtype=torch.float16)
    reference = x.float() @ dense.T
    scale = reference.abs().mean()
    for kernel in (int4pack_gemm, int4pack_gemm_v2):
        assert float((kernel(x, packed, qparams).float() - reference).abs().max() / scale) < 1e-2
    tinygemm = torch.ops.aten._weight_int4pack_mm(x.to(torch.bfloat16), packed, 128, qparams).float()
    # tinygemm rounds activations to bf16; the Triton path keeps FP16 and is closer.
    assert float((tinygemm - reference).abs().max() / scale) < 6e-2


@cuda
@requires_cuda
def test_awq_linear_routes_by_row_count() -> None:
    from engine.quantization.awq_tinygemm import AWQTinyGemmLinear, install_int4pack_prefill

    packed, qparams, dense = _packed(512, 1024)
    layer = AWQTinyGemmLinear(1024, 512, device="cuda")
    layer.packed_weight.copy_(packed)
    layer.scale_and_zero.copy_(qparams)
    x = torch.randn(2, 70, 1024, device="cuda", dtype=torch.float16)
    baseline = layer(x).float()
    assert install_int4pack_prefill(layer, 64) == 1
    routed = layer(x).float()
    reference = x.float() @ dense.T
    assert float((routed - reference).abs().max()) <= float((baseline - reference).abs().max())
    install_int4pack_prefill(layer, None)
    torch.testing.assert_close(layer(x).float(), baseline)
