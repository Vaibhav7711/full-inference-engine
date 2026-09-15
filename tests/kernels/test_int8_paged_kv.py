"""CUDA gates for kernel-native INT8 paged decode KV."""

from __future__ import annotations

import pytest
import torch


cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
def test_int8_decode_writer_matches_per_vector_reference() -> None:
    from engine.kernels.int8_paged_kv import write_decode_int8_kv

    torch.manual_seed(61)
    batch, heads, dim = 3, 8, 128
    block_size, blocks = 16, 12
    key = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    tables = torch.tensor([[9, 2, 7], [1, 11, 4], [6, 3, 10]], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, 15, 16], device="cuda", dtype=torch.int32)
    key_pages = torch.zeros(blocks, block_size, heads, dim, device="cuda", dtype=torch.int8)
    value_pages = torch.zeros_like(key_pages)
    key_scales = torch.zeros(blocks, block_size, heads, device="cuda", dtype=torch.float16)
    value_scales = torch.zeros_like(key_scales)

    write_decode_int8_kv(
        key, value, key_pages, value_pages, key_scales, value_scales, tables, positions,
    )
    torch.cuda.synchronize()
    for row, position in enumerate(positions.tolist()):
        logical, offset = divmod(position, block_size)
        physical = int(tables[row, logical])
        for source, pages, scales in ((key[row, :, 0], key_pages, key_scales),
                                      (value[row, :, 0], value_pages, value_scales)):
            expected_scale = source.float().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
            expected = torch.round(source.float() / expected_scale[:, None]).clamp(-127, 127).to(torch.int8)
            torch.testing.assert_close(scales[physical, offset], expected_scale.to(torch.float16), atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(pages[physical, offset], expected, atol=0, rtol=0)


@cuda
@requires_cuda
def test_int8_prefill_writer_maps_chunks_and_ignores_padding() -> None:
    from engine.kernels.int8_paged_kv import write_prefill_int8_kv_batched

    torch.manual_seed(63)
    batch, heads, padded_length, dim = 2, 4, 9, 128
    block_size, blocks = 4, 12
    starts = torch.tensor([3, 8], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([6, 9], device="cuda", dtype=torch.int32)
    tables = torch.tensor([[7, 2, 10, 1, 0], [5, 11, 3, 9, 6]], device="cuda", dtype=torch.int32)
    key = torch.randn(batch, heads, padded_length, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    key_pages = torch.zeros(blocks, block_size, heads, dim, device="cuda", dtype=torch.int8)
    value_pages = torch.zeros_like(key_pages)
    key_scales = torch.zeros(blocks, block_size, heads, device="cuda", dtype=torch.float16)
    value_scales = torch.zeros_like(key_scales)

    write_prefill_int8_kv_batched(
        key, value, key_pages, value_pages, key_scales, value_scales, tables, lengths, starts,
    )
    torch.cuda.synchronize()
    for row, length in enumerate(lengths.tolist()):
        for token in range(length):
            logical, offset = divmod(int(starts[row]) + token, block_size)
            physical = int(tables[row, logical])
            for source, pages, scales in ((key[row, :, token], key_pages, key_scales),
                                          (value[row, :, token], value_pages, value_scales)):
                expected_scale = source.float().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
                expected = torch.round(source.float() / expected_scale[:, None]).clamp(-127, 127).to(torch.int8)
                torch.testing.assert_close(scales[physical, offset], expected_scale.to(torch.float16), atol=1e-4, rtol=1e-3)
                # PyTorch uses round-to-even at exact half steps; the Triton writer
                # rounds halves away from zero. Both are nearest-integer quantizers,
                # so a tie may differ by exactly one INT8 level.
                quantization_delta = (pages[physical, offset].to(torch.int16) - expected.to(torch.int16)).abs()
                assert int(quantization_delta.max()) <= 1


@cuda
@requires_cuda
@pytest.mark.parametrize("sequence_length", [64, 256, 1024])
def test_int8_paged_decode_stays_close_to_fp16(sequence_length: int) -> None:
    from engine.kernels.int8_paged_kv import paged_decode_batched_int8
    from engine.kernels.paged_decode_batched import paged_decode_batched

    torch.manual_seed(67 + sequence_length)
    batch, q_heads, kv_heads, dim, block_size = 4, 16, 8, 128, 16
    blocks_per_sequence = (sequence_length + block_size - 1) // block_size
    total_blocks = batch * blocks_per_sequence
    # Reverse physical assignment verifies that the INT8 path uses the block table.
    tables = torch.arange(total_blocks - 1, -1, -1, device="cuda", dtype=torch.int32).view(batch, -1)
    lengths = torch.full((batch,), sequence_length, device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, 1, dim, device="cuda", dtype=torch.float16)
    key_fp16 = torch.randn(total_blocks, block_size, kv_heads, dim, device="cuda", dtype=torch.float16)
    value_fp16 = torch.randn_like(key_fp16)
    key_scale = key_fp16.float().abs().amax(dim=-1).clamp_min(1e-8).div(127).to(torch.float16)
    value_scale = value_fp16.float().abs().amax(dim=-1).clamp_min(1e-8).div(127).to(torch.float16)
    key_int8 = torch.round(key_fp16.float() / key_scale[..., None]).clamp(-127, 127).to(torch.int8)
    value_int8 = torch.round(value_fp16.float() / value_scale[..., None]).clamp(-127, 127).to(torch.int8)

    fp16 = paged_decode_batched(query, key_fp16, value_fp16, tables, lengths, block_n=128)
    int8 = paged_decode_batched_int8(
        query, key_int8, value_int8, key_scale, value_scale, tables, lengths, block_n=128,
    )
    relative_error = (int8.float() - fp16.float()).abs().mean() / fp16.float().abs().mean().clamp_min(1e-5)
    assert relative_error.item() < 0.03, f"INT8 paged attention relative error {relative_error.item():.4f}"


@cuda
@requires_cuda
def test_int8_paged_prefill_stays_close_to_fp16() -> None:
    from engine.kernels.int8_paged_kv import paged_prefill_int8
    from engine.kernels.paged_prefill import paged_prefill

    torch.manual_seed(71)
    batch, q_heads, kv_heads, dim, block_size, chunk = 2, 16, 8, 128, 16, 32
    starts = torch.tensor([64, 80], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([chunk, 21], device="cuda", dtype=torch.int32)
    max_sequence = int((starts + lengths).max())
    total_blocks = 16
    tables = torch.tensor([[9, 2, 7, 1, 12, 3, 4], [6, 11, 0, 14, 5, 13, 8]], device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, chunk, dim, device="cuda", dtype=torch.float16)
    key_fp16 = torch.randn(total_blocks, block_size, kv_heads, dim, device="cuda", dtype=torch.float16)
    value_fp16 = torch.randn_like(key_fp16)
    key_scale = key_fp16.float().abs().amax(dim=-1).clamp_min(1e-8).div(127).to(torch.float16)
    value_scale = value_fp16.float().abs().amax(dim=-1).clamp_min(1e-8).div(127).to(torch.float16)
    key_int8 = torch.round(key_fp16.float() / key_scale[..., None]).clamp(-127, 127).to(torch.int8)
    value_int8 = torch.round(value_fp16.float() / value_scale[..., None]).clamp(-127, 127).to(torch.int8)
    assert max_sequence <= tables.shape[1] * block_size

    fp16 = paged_prefill(query, key_fp16, value_fp16, tables, starts, lengths)
    int8 = paged_prefill_int8(query, key_int8, value_int8, key_scale, value_scale, tables, starts, lengths)
    valid = torch.cat([int8[row, :, :length].reshape(-1) for row, length in enumerate(lengths.tolist())])
    valid_reference = torch.cat([fp16[row, :, :length].reshape(-1) for row, length in enumerate(lengths.tolist())])
    relative_error = (valid.float() - valid_reference.float()).abs().mean() / valid_reference.float().abs().mean().clamp_min(1e-5)
    assert relative_error.item() < 0.03, f"INT8 paged prefill relative error {relative_error.item():.4f}"
