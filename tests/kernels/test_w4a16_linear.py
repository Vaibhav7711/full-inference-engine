"""W4A16: the pack/unpack contract on CPU, the kernel against it on CUDA.

The format module is Triton-free, so the packing layout the kernel assumes - split-half
nibbles within a group, offset 8, symmetric [-7, 7] - is pinned here before any launch.
"""

from __future__ import annotations

import pytest
import torch

from engine.kernels import w4a16_format as w4

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_pack_unpack_roundtrip_is_exact_for_in_range_integers():
    torch.manual_seed(3)
    n, k, g = 6, 512, 128
    # Build weights that are exact multiples of a per-group scale so quantisation is lossless.
    scales = torch.rand(n, k // g) + 0.5
    q = torch.randint(-7, 8, (n, k // g, g)).float()
    weight = (q * scales[..., None]).reshape(n, k)
    packed, out_scales = w4.quantize_weight_w4_grouped(weight, g)
    assert packed.shape == (n, k // 2) and packed.dtype is torch.uint8
    assert out_scales.shape == (n, k // g) and out_scales.dtype is torch.float16
    restored = w4.unpack_w4_reference(packed, out_scales, k, g).float()
    torch.testing.assert_close(restored, weight.to(torch.float16).float(), rtol=2e-3, atol=2e-3)


def test_split_half_layout_puts_input_j_and_j_plus_half_in_one_byte():
    n, k, g = 1, 256, 128
    half = g // 2
    weight = torch.zeros(n, k)
    # Group 0: input 5 -> +7 (nibble 15), input 5 + half -> -7 (nibble 1). Scale is 1.
    weight[0, 5] = 7.0
    weight[0, 5 + half] = -7.0
    weight[0, 0] = 7.0            # pins the group scale at 7/7 = 1
    packed, scales = w4.quantize_weight_w4_grouped(weight, g)
    assert float(scales[0, 0]) == pytest.approx(1.0)
    byte = int(packed[0, 5])
    assert byte & 0x0F == 15 and byte >> 4 == 1
    # Byte 0 of group 1 is at offset half; an all-zero group packs to 0x88 (8 | 8 << 4).
    assert int(packed[0, half]) == 0x88


def test_quantisation_error_is_within_half_a_step_per_group():
    torch.manual_seed(11)
    n, k, g = 8, 1024, 128
    weight = torch.randn(n, k) * torch.linspace(0.1, 4.0, n)[:, None]
    packed, scales = w4.quantize_weight_w4_grouped(weight, g)
    restored = w4.unpack_w4_reference(packed, scales, k, g).float()
    error = (restored - weight).abs().reshape(n, k // g, g).amax(dim=-1)
    # Half a quantisation step, plus fp16 rounding of the scale itself.
    assert torch.all(error <= scales.float() / 2 * 1.01 + 1e-3)


def test_rejects_shapes_the_kernel_cannot_tile():
    with pytest.raises(ValueError):
        w4.quantize_weight_w4_grouped(torch.zeros(4, 100), 128)     # K not a multiple of G
    with pytest.raises(ValueError):
        w4.quantize_weight_w4_grouped(torch.zeros(4, 128), 127)     # odd group


def test_weight_bytes_arithmetic():
    bytes_ = w4.weight_bytes(in_features=1024, out_features=3072)
    assert bytes_["fp16"] == 2 * 1024 * 3072
    assert bytes_["w4a16"] < bytes_["w8a16"] / 2 + 3072 * 2 * 8
    assert bytes_["w4a16"] * 3.5 < bytes_["fp16"]     # > 3.5x fewer bytes incl. scales


@cuda
@requires_cuda
@pytest.mark.parametrize("shape", [(16, 1024, 1024), (16, 1024, 3072), (1, 1024, 1024), (5, 2048, 896)])
def test_w4a16_linear_matches_dequantized_reference(shape: tuple[int, int, int]) -> None:
    kernels = pytest.importorskip("engine.kernels.w4a16_linear")

    torch.manual_seed(sum(shape))
    rows, width, outputs = shape
    inputs = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    weight = torch.randn(outputs, width, device="cuda", dtype=torch.float16)
    bias = torch.randn(outputs, device="cuda", dtype=torch.float16)
    packed, scales = w4.quantize_weight_w4_grouped(weight)
    dequantised = w4.unpack_w4_reference(packed, scales, width)
    reference = torch.nn.functional.linear(inputs, dequantised, bias)
    actual = kernels.w4a16_linear(inputs, packed, scales, bias)
    torch.testing.assert_close(actual, reference, rtol=5e-3, atol=5e-3)


@cuda
@requires_cuda
def test_w4a16_linear_without_bias_and_with_a_ragged_m() -> None:
    kernels = pytest.importorskip("engine.kernels.w4a16_linear")

    torch.manual_seed(2)
    inputs = torch.randn(13, 512, device="cuda", dtype=torch.float16)
    weight = torch.randn(200, 512, device="cuda", dtype=torch.float16)   # N not a tile multiple
    packed, scales = w4.quantize_weight_w4_grouped(weight)
    reference = inputs @ w4.unpack_w4_reference(packed, scales, 512).T
    actual = kernels.w4a16_linear(inputs, packed, scales)
    torch.testing.assert_close(actual, reference, rtol=5e-3, atol=5e-3)
