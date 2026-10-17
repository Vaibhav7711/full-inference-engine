"""Correctness tests for the experimental direct-from-packed AWQ projection."""

import pytest
import torch

from engine.kernels.awq_gemv import awq_gemv
from engine.quantization.awq_tinygemm import AWQTinyGemmLinear, repack_awq_to_tinygemm


def _pack_awq(logical: torch.Tensor) -> torch.Tensor:
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=logical.device)
    stored = logical.reshape(logical.shape[0], -1, 8)[:, :, order]
    shifts = (4 * torch.arange(8, device=logical.device)).to(torch.int32)
    return torch.sum(stored.to(torch.int32) << shifts, dim=-1).to(torch.int32)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch,splits", [(1, 1), (1, 2), (1, 4), (4, 2)])
def test_awq_gemv_matches_explicit_dequant(batch: int, splits: int) -> None:
    torch.manual_seed(146)
    k, n = 256, 128
    weight = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32)
    zeros = torch.randint(0, 16, (k // 128, n), device="cuda", dtype=torch.int32)
    scales = torch.rand((k // 128, n), device="cuda", dtype=torch.float16) * 0.1
    x = torch.randn((batch, k), device="cuda", dtype=torch.float16)
    qweight = _pack_awq(weight)
    qzeros = _pack_awq(zeros)
    dequant = ((weight - zeros.repeat_interleave(128, dim=0)).float() *
               scales.repeat_interleave(128, dim=0).float()).to(torch.float16)
    reference = x @ dequant
    actual = awq_gemv(x, qweight, qzeros, scales, block_k=64, block_n=32,
                      splits=splits)
    torch.testing.assert_close(actual, reference, atol=0.03, rtol=0.003)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_awq_repack_tinygemm_matches_explicit_dequant_for_3d_input() -> None:
    torch.manual_seed(147)
    k, n = 256, 128
    weight = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32)
    zeros = torch.randint(0, 16, (k // 128, n), device="cuda", dtype=torch.int32)
    scales = torch.rand((k // 128, n), device="cuda", dtype=torch.float16) * 0.1
    qweight, qzeros = _pack_awq(weight), _pack_awq(zeros)
    packed, qparams = repack_awq_to_tinygemm(qweight, qzeros, scales)
    module = AWQTinyGemmLinear(k, n, device="cuda")
    module.packed_weight.copy_(packed)
    module.scale_and_zero.copy_(qparams)
    inputs = torch.randn((2, 3, k), device="cuda", dtype=torch.float16)
    dequant = ((weight - zeros.repeat_interleave(128, dim=0)).float() *
               scales.repeat_interleave(128, dim=0).float()).to(torch.float16)
    actual, reference = module(inputs), inputs @ dequant
    relative_mean = (actual.float() - reference.float()).abs().mean() / reference.float().abs().mean()
    assert float(relative_mean) < 0.005
    assert float((actual - reference).abs().max()) < 0.15
