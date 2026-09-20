"""SDPA chunked prefill: mask and gather semantics (CPU), engine tokens (CUDA)."""

from __future__ import annotations

import pytest
import torch

from engine.kernels.sdpa_prefill import chunk_causal_mask, gather_pages, sdpa_paged_prefill

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_chunk_causal_mask_sees_prefix_and_own_causal_region_only() -> None:
    starts = torch.tensor([4, 0], dtype=torch.int32)
    chunks = torch.tensor([3, 2], dtype=torch.int32)
    mask = chunk_causal_mask(starts, chunks, query_len=3, total_len=7)
    assert mask.shape == (2, 1, 3, 7)
    # Row 0: chunk starts at 4; query j sees positions <= 4 + j.
    assert mask[0, 0].tolist() == [
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    # Row 1: chunk of 2 from position 0; row 2 is padding and sees key 0 only.
    assert mask[1, 0].tolist() == [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ]
    assert bool(mask.any(dim=-1).all()), "every row must keep at least one key (no NaN softmax)"


def test_gather_pages_follows_the_block_table_and_clamps_padding() -> None:
    pool = torch.arange(6 * 4 * 1 * 2, dtype=torch.float32).view(6, 4, 1, 2)  # 6 pages x 4 slots
    tables = torch.tensor([[5, 2, -1], [1, -1, -1]], dtype=torch.int32)
    dense = gather_pages(pool, tables, total_len=6)                       # [B, H, 6, D]
    assert dense.shape == (2, 1, 6, 2)
    assert torch.equal(dense[0, 0, :4], pool[5, :, 0])
    assert torch.equal(dense[0, 0, 4:6], pool[2, :2, 0])
    assert torch.equal(dense[1, 0, :4], pool[1, :, 0])
    assert torch.equal(dense[1, 0, 4:6], pool[0, :2, 0])                  # -1 clamped to page 0


def test_sdpa_paged_prefill_matches_dense_reference_on_cpu() -> None:
    torch.manual_seed(3)
    batch, q_heads, kv_heads, head_dim, block = 2, 4, 2, 8, 4
    starts = torch.tensor([8, 3], dtype=torch.int32)
    chunks = torch.tensor([4, 2], dtype=torch.int32)
    query_len, total = 4, 12
    pages = 8
    key_pool = torch.randn(pages, block, kv_heads, head_dim)
    value_pool = torch.randn_like(key_pool)
    tables = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]], dtype=torch.int32)
    query = torch.randn(batch, q_heads, query_len, head_dim)
    out = sdpa_paged_prefill(query, key_pool, value_pool, tables, starts, chunks, total_len=total)

    for b in range(batch):
        keys = gather_pages(key_pool, tables, total)[b].repeat_interleave(q_heads // kv_heads, 0)
        values = gather_pages(value_pool, tables, total)[b].repeat_interleave(q_heads // kv_heads, 0)
        for j in range(int(chunks[b])):
            visible = int(starts[b]) + j + 1
            scores = torch.einsum("hd,htd->ht", query[b, :, j], keys[:, :visible]) * head_dim ** -0.5
            probs = torch.softmax(scores, dim=-1)
            expected = torch.einsum("ht,htd->hd", probs, values[:, :visible])
            torch.testing.assert_close(out[b, :, j], expected, rtol=1e-4, atol=1e-5)


@cuda
@requires_cuda
def test_sdpa_chunked_prefill_matches_per_token_kernel_output() -> None:
    from engine.kernels.paged_prefill import paged_prefill

    torch.manual_seed(5)
    batch, q_heads, kv_heads, head_dim = 3, 16, 8, 128
    key_pool = torch.randn(64, 16, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    value_pool = torch.randn_like(key_pool)
    tables = torch.arange(64, device="cuda", dtype=torch.int32).view(batch, -1)[:, :20].contiguous()
    starts = torch.tensor([200, 0, 37], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([64, 64, 9], device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, 64, head_dim, device="cuda", dtype=torch.float16)
    reference = paged_prefill(query, key_pool, value_pool, tables, starts, chunks)
    actual = sdpa_paged_prefill(query, key_pool, value_pool, tables, starts, chunks, total_len=264)
    for b in range(batch):
        n = int(chunks[b])
        torch.testing.assert_close(actual[b, :, :n], reference[b, :, :n], rtol=2e-2, atol=2e-2)
