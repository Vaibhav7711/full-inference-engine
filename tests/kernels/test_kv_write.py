from __future__ import annotations

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_decode_kv_write_matches_reference() -> None:
    from engine.kernels.kv_write import write_decode_kv

    torch.manual_seed(11)
    batch_size, heads, head_dim = 5, 8, 128
    block_size, physical_blocks, table_width = 16, 24, 8
    key = torch.randn(
        batch_size, heads, 1, head_dim, device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    block_tables = torch.tensor(
        [
            [9, 2, 11, 4, 1, 7, 6, 3],
            [5, 13, 8, 16, 0, 12, 10, 14],
            [17, 19, 21, 23, 18, 20, 22, 15],
            [3, 6, 9, 12, 15, 18, 21, 0],
            [1, 4, 7, 10, 13, 16, 19, 22],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert block_tables.shape == (batch_size, table_width)
    # Exercise first/last offsets and both sides of multiple block boundaries.
    seq_lens = torch.tensor([0, 15, 16, 31, 65], dtype=torch.int32, device="cuda")
    key_pool = torch.zeros(
        physical_blocks, block_size, heads, head_dim, device="cuda", dtype=torch.float16
    )
    value_pool = torch.zeros_like(key_pool)

    write_decode_kv(key, value, key_pool, value_pool, block_tables, seq_lens)
    torch.cuda.synchronize()

    for batch in range(batch_size):
        position = int(seq_lens[batch].item())
        logical_block, offset = divmod(position, block_size)
        physical_block = int(block_tables[batch, logical_block].item())
        assert torch.equal(key_pool[physical_block, offset], key[batch, :, 0])
        assert torch.equal(value_pool[physical_block, offset], value[batch, :, 0])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_prefill_write_ignores_padding_and_maps_each_request() -> None:
    from engine.kernels.kv_write import write_prefill_kv_batched

    torch.manual_seed(41)
    batch, heads, padded_length, head_dim = 3, 8, 33, 128
    block_size, physical_blocks = 16, 20
    lengths = torch.tensor([7, 16, 33], dtype=torch.int32, device="cuda")
    tables = torch.tensor(
        [[9, -1, -1], [2, -1, -1], [11, 4, 17]],
        dtype=torch.int32,
        device="cuda",
    )
    key = torch.randn(
        batch, heads, padded_length, head_dim, device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    key_pool = torch.zeros(
        physical_blocks, block_size, heads, head_dim, device="cuda", dtype=torch.float16
    )
    value_pool = torch.zeros_like(key_pool)

    write_prefill_kv_batched(key, value, key_pool, value_pool, tables, lengths)
    torch.cuda.synchronize()

    for row, length in enumerate(lengths.tolist()):
        for token in range(length):
            logical_block, offset = divmod(token, block_size)
            physical_block = int(tables[row, logical_block].item())
            assert torch.equal(key_pool[physical_block, offset], key[row, :, token])
            assert torch.equal(value_pool[physical_block, offset], value[row, :, token])
    # Padding for the short rows must never be written through their -1 sentinels.
    assert torch.count_nonzero(key_pool[-1]) == 0
    assert torch.count_nonzero(value_pool[-1]) == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_prefill_write_supports_nonzero_chunk_starts() -> None:
    from engine.kernels.kv_write import write_prefill_kv_batched

    torch.manual_seed(53)
    batch, heads, padded_length, head_dim = 2, 4, 9, 64
    block_size = 4
    starts = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([6, 9], dtype=torch.int32, device="cuda")
    tables = torch.tensor([[7, 2, 10, 1, 0], [5, 11, 3, 9, 6]], device="cuda")
    key = torch.randn(batch, heads, padded_length, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    key_pool = torch.zeros(12, block_size, heads, head_dim, device="cuda", dtype=torch.float16)
    value_pool = torch.zeros_like(key_pool)

    write_prefill_kv_batched(
        key, value, key_pool, value_pool, tables, lengths, starts
    )
    torch.cuda.synchronize()
    for row in range(batch):
        for token in range(int(lengths[row])):
            position = int(starts[row]) + token
            logical_block, offset = divmod(position, block_size)
            physical_block = int(tables[row, logical_block])
            assert torch.equal(key_pool[physical_block, offset], key[row, :, token])
            assert torch.equal(value_pool[physical_block, offset], value[row, :, token])
