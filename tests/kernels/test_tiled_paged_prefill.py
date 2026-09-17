"""The tiled prefill kernel must be a drop-in replacement, not merely a faster one.

Correctness is checked three ways, because a fast kernel that shifts one logit is worse
than a slow correct one: against a dense SDPA reference built from the same pages, against
the existing per-token kernel it replaces, and on the edges (ragged chunks, padded rows,
non-contiguous page tables, group-query head mapping).
"""

from __future__ import annotations

import math

import pytest
import torch

from engine.kernels.paged_prefill import paged_prefill
from engine.kernels.tiled_paged_prefill import tiled_paged_prefill

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HEAD_DIM = 128
BLOCK_SIZE = 16


def _build(batch, q_heads, kv_heads, starts, chunks, num_pages=512, seed=0, shuffle=True):
    """A paged KV pool with deliberately scattered page assignments."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    max_len = max(s + c for s, c in zip(starts, chunks))
    blocks_needed = -(-max_len // BLOCK_SIZE)
    key_pages = torch.randn(num_pages, BLOCK_SIZE, kv_heads, HEAD_DIM,
                            device="cuda", dtype=torch.float16, generator=generator)
    value_pages = torch.randn_like(key_pages)
    tables = torch.zeros(batch, blocks_needed + 2, device="cuda", dtype=torch.int32)
    pool = torch.randperm(num_pages, generator=generator, device="cuda")
    cursor = 0
    for row in range(batch):
        count = blocks_needed + 2
        chosen = pool[cursor:cursor + count] if shuffle else torch.arange(
            cursor, cursor + count, device="cuda")
        cursor += count
        tables[row, :count] = chosen.to(torch.int32)
    query_len = max(chunks)
    query = torch.randn(batch, q_heads, query_len, HEAD_DIM,
                        device="cuda", dtype=torch.float16, generator=generator)
    start_t = torch.tensor(starts, device="cuda", dtype=torch.int32)
    chunk_t = torch.tensor(chunks, device="cuda", dtype=torch.int32)
    return query, key_pages, value_pages, tables, start_t, chunk_t


def _dense_reference(query, key_pages, value_pages, tables, starts, chunks):
    """Gather pages into dense tensors and run plain masked softmax attention in fp32."""
    batch, q_heads, query_len, head_dim = query.shape
    kv_heads = key_pages.shape[2]
    group = q_heads // kv_heads
    out = torch.zeros_like(query, dtype=torch.float32)
    for row in range(batch):
        start, chunk = int(starts[row]), int(chunks[row])
        total = start + chunk
        keys = torch.empty(total, kv_heads, head_dim, device=query.device, dtype=torch.float32)
        values = torch.empty_like(keys)
        for position in range(total):
            page = int(tables[row, position // BLOCK_SIZE])
            offset = position % BLOCK_SIZE
            keys[position] = key_pages[page, offset].float()
            values[position] = value_pages[page, offset].float()
        for head in range(q_heads):
            kv_head = head // group
            q = query[row, head, :chunk].float()
            scores = q @ keys[:, kv_head].T * (head_dim ** -0.5)
            positions = torch.arange(total, device=query.device)
            q_positions = start + torch.arange(chunk, device=query.device)
            scores = scores.masked_fill(positions[None, :] > q_positions[:, None], -math.inf)
            out[row, head, :chunk] = torch.softmax(scores, dim=-1) @ values[:, kv_head]
    return out


@cuda
@requires_cuda
@pytest.mark.parametrize("starts,chunks", [
    ([0], [64]),                 # fresh prompt, one full tile
    ([0], [17]),                 # ragged chunk shorter than a tile
    ([128], [128]),              # resumed chunk with a real prefix
    ([37], [96]),                # prefix not aligned to a page boundary
    ([0, 64, 200], [128, 96, 33]),   # mixed batch, padded rows
])
def test_matches_a_dense_fp32_reference(starts, chunks):
    query, kp, vp, tables, start_t, chunk_t = _build(
        len(starts), 16, 8, starts, chunks, seed=len(starts))
    got = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t)
    want = _dense_reference(query, kp, vp, tables, start_t, chunk_t)
    for row, chunk in enumerate(chunks):
        torch.testing.assert_close(
            got[row, :, :chunk].float(), want[row, :, :chunk], rtol=2e-2, atol=2e-2,
        )


@cuda
@requires_cuda
@pytest.mark.parametrize("starts,chunks", [
    ([0], [128]), ([256], [128]), ([0, 512], [64, 128]), ([1000], [200]),
])
def test_agrees_with_the_kernel_it_replaces(starts, chunks):
    query, kp, vp, tables, start_t, chunk_t = _build(
        len(starts), 16, 8, starts, chunks, seed=7)
    old = paged_prefill(query, kp, vp, tables, start_t, chunk_t)
    new = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t)
    for row, chunk in enumerate(chunks):
        torch.testing.assert_close(
            new[row, :, :chunk].float(), old[row, :, :chunk].float(),
            rtol=2e-2, atol=2e-2,
        )


@cuda
@requires_cuda
def test_padded_rows_are_zero_and_never_nan():
    """A fully masked row must not produce NaN through the online-softmax rescale."""
    query, kp, vp, tables, start_t, chunk_t = _build(2, 16, 8, [0, 0], [128, 8], seed=3)
    out = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t)
    assert torch.isfinite(out).all(), "kernel produced NaN or Inf"
    assert torch.count_nonzero(out[1, :, 8:]) == 0, "padded rows must be zero"


@cuda
@requires_cuda
@pytest.mark.parametrize("block_m,block_n", [(16, 32), (32, 64), (64, 64), (64, 32), (128, 64)])
def test_tile_shape_does_not_change_the_result(block_m, block_n):
    query, kp, vp, tables, start_t, chunk_t = _build(2, 16, 8, [0, 300], [128, 128], seed=5)
    baseline = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t,
                                   block_m=64, block_n=64)
    variant = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t,
                                  block_m=block_m, block_n=block_n)
    torch.testing.assert_close(variant.float(), baseline.float(), rtol=2e-2, atol=2e-2)


@cuda
@requires_cuda
def test_fp32_pv_path_agrees_with_the_tensor_core_path():
    query, kp, vp, tables, start_t, chunk_t = _build(1, 16, 8, [256], [128], seed=9)
    fast = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t)
    exact = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t, pv_in_fp32=True)
    torch.testing.assert_close(fast.float(), exact.float(), rtol=3e-2, atol=3e-2)


@cuda
@requires_cuda
def test_multi_query_attention_head_grouping():
    """kv_heads == 1 and kv_heads == q_heads are the geometry edges."""
    for kv_heads in (1, 16):
        query, kp, vp, tables, start_t, chunk_t = _build(
            1, 16, kv_heads, [64], [96], seed=11)
        got = tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t)
        want = _dense_reference(query, kp, vp, tables, start_t, chunk_t)
        torch.testing.assert_close(got[0, :, :96].float(), want[0, :, :96],
                                   rtol=2e-2, atol=2e-2)


@cuda
@requires_cuda
def test_rejects_tiles_below_the_tl_dot_minimum():
    query, kp, vp, tables, start_t, chunk_t = _build(1, 16, 8, [0], [64], seed=1)
    with pytest.raises(ValueError):
        tiled_paged_prefill(query, kp, vp, tables, start_t, chunk_t, block_m=8)
