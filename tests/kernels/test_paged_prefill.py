from __future__ import annotations

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunked_paged_prefill_matches_materialized_reference() -> None:
    from engine.kernels.paged_prefill import paged_prefill

    torch.manual_seed(67)
    batch, q_heads, kv_heads, query_len, head_dim = 3, 8, 2, 7, 64
    block_size, num_blocks = 4, 24
    starts = torch.tensor([0, 3, 9], dtype=torch.int32, device="cuda")
    chunks = torch.tensor([7, 5, 2], dtype=torch.int32, device="cuda")
    tables = torch.tensor(
        [[23, 4, 17, 8], [2, 21, 6, 13], [19, 1, 15, 10]],
        dtype=torch.int32, device="cuda",
    )
    key_pages = torch.randn(
        num_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    value_pages = torch.randn_like(key_pages)
    query = torch.randn(
        batch, q_heads, query_len, head_dim, device="cuda", dtype=torch.float16
    )

    actual = paged_prefill(
        query, key_pages, value_pages, tables, starts, chunks
    )
    expected = torch.zeros_like(actual)
    repetitions = q_heads // kv_heads
    scale = head_dim ** -0.5
    for row in range(batch):
        for query_token in range(int(chunks[row])):
            kv_len = int(starts[row]) + query_token + 1
            logical = []
            for position in range(kv_len):
                block, offset = divmod(position, block_size)
                physical = int(tables[row, block])
                logical.append((physical, offset))
            keys = torch.stack([key_pages[p, o] for p, o in logical])
            values = torch.stack([value_pages[p, o] for p, o in logical])
            keys = keys.repeat_interleave(repetitions, dim=1).transpose(0, 1)
            values = values.repeat_interleave(repetitions, dim=1).transpose(0, 1)
            scores = torch.einsum("hd,hkd->hk", query[row, :, query_token].float(), keys.float())
            probabilities = torch.softmax(scores * scale, dim=-1)
            expected[row, :, query_token] = torch.einsum(
                "hk,hkd->hd", probabilities, values.float()
            ).to(expected.dtype)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(actual[1, :, 5:]) == 0
    assert torch.count_nonzero(actual[2, :, 2:]) == 0
