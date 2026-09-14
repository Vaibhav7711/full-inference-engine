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
