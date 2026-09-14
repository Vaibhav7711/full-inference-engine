from __future__ import annotations

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sequence_length", [1, 15, 16, 17, 41, 64])
def test_fused_kv_write_matches_reference_scatter(sequence_length: int) -> None:
    from engine.kernels.kv_write import write_paged_kv

    torch.manual_seed(7)
    heads, head_dim, block_size, physical_blocks = 8, 128, 16, 12
    key = torch.randn(
        1, heads, sequence_length, head_dim, device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    blocks_needed = (sequence_length + block_size - 1) // block_size
    block_table = torch.tensor(
        [9, 2, 11, 4][:blocks_needed], dtype=torch.int32, device="cuda"
    )
    key_pool = torch.zeros(
        physical_blocks, block_size, heads, head_dim, device="cuda", dtype=torch.float16
    )
    value_pool = torch.zeros_like(key_pool)

    write_paged_kv(key, value, key_pool, value_pool, block_table)
    torch.cuda.synchronize()

    for token in range(sequence_length):
        logical_block, offset = divmod(token, block_size)
        physical_block = int(block_table[logical_block].item())
        assert torch.equal(key_pool[physical_block, offset], key[0, :, token])
        assert torch.equal(value_pool[physical_block, offset], value[0, :, token])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_write_rejects_short_block_table() -> None:
    from engine.kernels.kv_write import write_paged_kv

    key = torch.zeros(1, 2, 17, 8, device="cuda", dtype=torch.float16)
    pool = torch.zeros(4, 16, 2, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="does not cover"):
        write_paged_kv(key, key, pool, pool.clone(), torch.tensor([0], device="cuda"))


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
