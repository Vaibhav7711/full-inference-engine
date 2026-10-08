"""FP8 (E4M3) paged KV: the reference quantiser on CPU, the kernels on CUDA.

The CPU tests pin the arithmetic contract the kernels are written to - scale, clamp,
round-to-nearest, no NaN - so a kernel bug on the card shows up as a disagreement with
something already known to be right, not as two unknowns. The CUDA tests compare each
kernel against the fp16 kernel it shadows, run over the *dequantised* pages, so the only
thing under test is the kernel's arithmetic and addressing, not the quantisation error.
"""

from __future__ import annotations

import pytest
import torch

# The contract is Triton-free; the kernels are imported inside the tests that launch
# them, so this module collects on a machine without Triton.
from engine.kernels import fp8_format as fp8
from engine.kernels.device import DeviceProfile

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_fp8_dtype = pytest.mark.skipif(fp8.FP8_DTYPE is None, reason="torch has no float8_e4m3fn")


def _profile(sm: int) -> DeviceProfile:
    major, minor = divmod(sm, 10)
    return DeviceProfile(name=f"sm_{sm}", capability=(major, minor), total_memory_gb=8.0,
                         multiprocessors=24, l2_cache_mb=24.0)


# ---------------------------------------------------------------------------
# CPU: the contract
# ---------------------------------------------------------------------------

def test_e4m3_max_matches_torch_finfo():
    if fp8.FP8_DTYPE is None:
        pytest.skip("torch has no float8_e4m3fn")
    assert fp8.E4M3_MAX == float(torch.finfo(fp8.FP8_DTYPE).max) == 448.0


@requires_fp8_dtype
def test_reference_roundtrip_error_is_bounded_by_half_an_ulp_of_the_vector_max():
    torch.manual_seed(7)
    x = torch.randn(4, 8, 128) * torch.tensor([0.01, 1.0, 30.0, 500.0])[:, None, None]
    q, scale = fp8.quantize_fp8_reference(x)
    assert q.dtype is fp8.FP8_DTYPE and scale.shape == (4, 8)
    deq = fp8.dequantize_fp8_reference(q, scale)
    # E4M3 keeps 3 mantissa bits: the largest element lands at 1.75 * 2^8 where a half
    # ulp is 16/448 = 3.6% of the vector max, and every smaller element's absolute error
    # is at most that. 1/16 of the max is a safe bound for all of them.
    error = (x - deq).abs().amax(dim=-1)
    assert torch.all(error <= x.abs().amax(dim=-1) / 16 + 1e-6)
    assert not torch.isnan(deq).any()


@requires_fp8_dtype
def test_reference_is_total_at_the_edges():
    zeros = torch.zeros(2, 3, 64)
    q, scale = fp8.quantize_fp8_reference(zeros)
    assert torch.all(scale == 1e-8) and torch.all(q.float() == 0)
    huge = torch.full((1, 1, 32), 65504.0)      # fp16 max
    q, scale = fp8.quantize_fp8_reference(huge)
    assert not torch.isnan(q.float()).any()
    assert torch.allclose(fp8.dequantize_fp8_reference(q, scale), huge, rtol=1e-3)
    # A vector whose max is negative scales by |max| and the sign survives the clamp.
    neg = -torch.rand(1, 1, 16) - 1.0
    q, scale = fp8.quantize_fp8_reference(neg)
    assert torch.all(q.float() <= 0)


def test_availability_is_gated_on_sm89():
    reason = fp8.unavailable_reason(_profile(75))
    assert reason is not None and "sm_89" in reason and "sm_75" in reason
    reason = fp8.unavailable_reason(_profile(86))
    assert reason is not None and "sm_86" in reason
    if fp8.FP8_DTYPE is not None:
        assert fp8.unavailable_reason(_profile(89)) is None
        assert fp8.unavailable_reason(_profile(90)) is None
        assert fp8.unavailable_reason(None) is None   # no device: nothing to refuse on


@requires_fp8_dtype
def test_wrappers_refuse_wrong_pool_dtype_before_launching():
    kernels = pytest.importorskip("engine.kernels.fp8_paged_kv")
    pages = torch.zeros(4, 16, 8, 128, dtype=torch.int8)     # INT8 handed to the FP8 path
    scales = torch.zeros(4, 16, 8, dtype=torch.float16)
    key = torch.zeros(1, 8, 1, 128, dtype=torch.float16)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        kernels.write_decode_fp8_kv(key, key, pages, pages, scales, scales,
                                    torch.zeros(1, 4, dtype=torch.int32), torch.zeros(1, dtype=torch.int32))


# ---------------------------------------------------------------------------
# CUDA: the kernels, against the fp16 kernels over dequantised pages
# ---------------------------------------------------------------------------

def _fp8_pages(blocks, block_size, heads, dim, device):
    """Random fp16 pages, their FP8 quantisation, and the fp16 pages that FP8 encodes."""
    fp16 = torch.randn(blocks, block_size, heads, dim, device=device, dtype=torch.float16)
    q, scale = fp8.quantize_fp8_reference(fp16)              # scale: [blocks, block, heads]
    exact = fp8.dequantize_fp8_reference(q, scale).to(torch.float16)
    return q.contiguous(), scale.to(torch.float16).contiguous(), exact.contiguous()


@cuda
@requires_cuda
@requires_fp8_dtype
def test_fp8_decode_writer_matches_the_reference_quantiser():
    torch.manual_seed(61)
    batch, heads, dim, block_size, blocks = 3, 8, 128, 16, 12
    key = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    tables = torch.tensor([[9, 2, 7], [1, 11, 4], [6, 3, 10]], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, 15, 16], device="cuda", dtype=torch.int32)
    key_pages = torch.zeros(blocks, block_size, heads, dim, device="cuda", dtype=fp8.FP8_DTYPE)
    value_pages = torch.zeros_like(key_pages)
    key_scales = torch.zeros(blocks, block_size, heads, device="cuda", dtype=torch.float16)
    value_scales = torch.zeros_like(key_scales)

    kernels = pytest.importorskip("engine.kernels.fp8_paged_kv")
    kernels.write_decode_fp8_kv(key, value, key_pages, value_pages, key_scales, value_scales,
                                tables, positions)
    torch.cuda.synchronize()
    for row, position in enumerate(positions.tolist()):
        logical, offset = divmod(position, block_size)
        physical = int(tables[row, logical])
        for source, pages, scales in ((key[row, :, 0], key_pages, key_scales),
                                      (value[row, :, 0], value_pages, value_scales)):
            expected_q, expected_scale = fp8.quantize_fp8_reference(source)
            torch.testing.assert_close(scales[physical, offset].float(), expected_scale,
                                       atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(pages[physical, offset].float(), expected_q.float(),
                                       atol=0, rtol=0)


@cuda
@requires_cuda
@requires_fp8_dtype
def test_fp8_prefill_writer_maps_chunks_and_ignores_padding():
    torch.manual_seed(63)
    batch, heads, padded, dim, block_size, blocks = 2, 4, 9, 128, 4, 12
    starts = torch.tensor([3, 8], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([6, 9], device="cuda", dtype=torch.int32)
    tables = torch.tensor([[7, 2, 10, 1, 0], [5, 11, 3, 9, 6]], device="cuda", dtype=torch.int32)
    key = torch.randn(batch, heads, padded, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    key_pages = torch.zeros(blocks, block_size, heads, dim, device="cuda", dtype=fp8.FP8_DTYPE)
    value_pages = torch.zeros_like(key_pages)
    key_scales = torch.zeros(blocks, block_size, heads, device="cuda", dtype=torch.float16)
    value_scales = torch.zeros_like(key_scales)

    kernels = pytest.importorskip("engine.kernels.fp8_paged_kv")
    kernels.write_prefill_fp8_kv_batched(key, value, key_pages, value_pages, key_scales,
                                         value_scales, tables, lengths, starts)
    torch.cuda.synchronize()
    written = set()
    for row, length in enumerate(lengths.tolist()):
        for token in range(length):
            logical, offset = divmod(int(starts[row]) + token, block_size)
            physical = int(tables[row, logical])
            written.add((physical, offset))
            expected_q, expected_scale = fp8.quantize_fp8_reference(key[row, :, token])
            torch.testing.assert_close(key_pages[physical, offset].float(), expected_q.float(),
                                       atol=0, rtol=0)
            torch.testing.assert_close(key_scales[physical, offset].float(), expected_scale,
                                       atol=1e-4, rtol=1e-3)
    # Padding past each chunk's length must not have touched the pool.
    for physical in range(blocks):
        for offset in range(block_size):
            if (physical, offset) not in written:
                assert torch.all(key_pages[physical, offset].float() == 0)


@cuda
@requires_cuda
@requires_fp8_dtype
@pytest.mark.parametrize("block_n", [64, 128])
def test_fp8_decode_matches_fp16_per_head_over_the_same_values(block_n):
    from engine.kernels.paged_decode_batched import paged_decode_batched

    torch.manual_seed(5)
    batch, q_heads, kv_heads, dim, block_size, blocks = 4, 16, 8, 128, 16, 40
    key_q, key_s, key_exact = _fp8_pages(blocks, block_size, kv_heads, dim, "cuda")
    value_q, value_s, value_exact = _fp8_pages(blocks, block_size, kv_heads, dim, "cuda")
    tables = torch.randperm(blocks, device="cuda")[: batch * 10].view(batch, 10).to(torch.int32)
    seq_lens = torch.tensor([1, 37, 128, 159], device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, 1, dim, device="cuda", dtype=torch.float16)

    kernels = pytest.importorskip("engine.kernels.fp8_paged_kv")
    out = kernels.paged_decode_batched_fp8(query, key_q, value_q, key_s, value_s, tables,
                                           seq_lens, block_n=block_n)
    reference = paged_decode_batched(query, key_exact, value_exact, tables, seq_lens,
                                     block_n=block_n)
    torch.cuda.synchronize()
    # Both kernels accumulate in fp32 over identical values; the FP8 one dequantises
    # with an fp16 scale, so allow fp16 rounding of the scale, nothing more.
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@cuda
@requires_cuda
@requires_fp8_dtype
def test_fp8_prefill_matches_per_token_fp16_over_the_same_values():
    from engine.kernels.paged_prefill import paged_prefill

    torch.manual_seed(9)
    batch, q_heads, kv_heads, dim, block_size, blocks, width = 2, 16, 8, 128, 16, 48, 32
    key_q, key_s, key_exact = _fp8_pages(blocks, block_size, kv_heads, dim, "cuda")
    value_q, value_s, value_exact = _fp8_pages(blocks, block_size, kv_heads, dim, "cuda")
    tables = torch.randperm(blocks, device="cuda")[: batch * 20].view(batch, 20).to(torch.int32)
    starts = torch.tensor([0, 200], device="cuda", dtype=torch.int32)
    chunk_lens = torch.tensor([width, 17], device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, width, dim, device="cuda", dtype=torch.float16)

    kernels = pytest.importorskip("engine.kernels.fp8_paged_kv")
    out = kernels.paged_prefill_fp8(query, key_q, value_q, key_s, value_s, tables, starts, chunk_lens)
    reference = paged_prefill(query, key_exact, value_exact, tables, starts, chunk_lens)
    torch.cuda.synchronize()
    valid = torch.arange(width, device="cuda")[None, :] < chunk_lens[:, None]   # [B, W]
    mask = valid[:, None, :, None].expand_as(out)
    torch.testing.assert_close(out[mask], reference[mask], atol=2e-2, rtol=2e-2)
