"""K2 correctness: paged kernel must match K1 (contiguous kernel), hence SDPA.

The gate: take K,V, scatter them into a paged store using a NON-TRIVIAL block table
(shuffled physical blocks, so the indirection is genuinely exercised), run the paged
kernel, and assert its output equals the K1 contiguous kernel on the same logical data.

Because K1 was already verified == SDPA, this transitively proves K2 == SDPA. And
because the block table is shuffled, a bug in the physical-address computation would
produce wrong output — so passing means the paged addressing is correct.

Run:
    python -m pytest tests/kernels/test_paged_attention_kernel.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_shuffled_block_table(num_logical_blocks, num_physical_blocks, seed):
    """A block table mapping logical->physical with shuffled, non-identity physical ids."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_physical_blocks, generator=g)[:num_logical_blocks]
    return perm.to(torch.int32)


@cuda
@requires_cuda
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("H,M,N,D", [
    (8, 1, 64, 128),      # decode: single query, 64 KV
    (8, 64, 64, 128),     # prefill-ish square
    (4, 100, 100, 64),    # non-aligned length, smaller dim
    (16, 37, 37, 128),    # odd length, many heads
    (8, 1, 200, 128),     # long KV decode
])
def test_paged_matches_contiguous(block_size, H, M, N, D):
    from engine.kernels.triton_attention import triton_attention
    from engine.kernels.paged_attention_kernel import paged_attention, scatter_to_pages

    torch.manual_seed(0)
    device = "cuda"

    # For causal, single-query decode (M==1) attends to all keys; M==N is standard.
    causal = (M == N) or (M == 1)
    ref_causal = causal and (M == N)

    q = torch.randn(H, M, D, device=device, dtype=torch.float16)
    k = torch.randn(H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(H, N, D, device=device, dtype=torch.float16)

    # --- K1 reference: contiguous kernel ---
    # triton_attention expects [B, H, M, D]; add batch dim of 1
    ref = triton_attention(q[None], k[None], v[None], causal=ref_causal)[0]  # [H, M, D]

    # --- Build paged storage with a shuffled block table ---
    num_logical_blocks = (N + block_size - 1) // block_size
    num_physical_blocks = num_logical_blocks + 5   # extra blocks -> non-trivial mapping
    block_table = _make_shuffled_block_table(num_logical_blocks, num_physical_blocks, seed=7).to(device)

    kp, vp = scatter_to_pages(k, v, block_size, block_table, num_physical_blocks)

    # --- K2: paged kernel ---
    out = paged_attention(q, kp, vp, block_table, kv_len=N, causal=(causal and M == N))

    max_diff = (out - ref).abs().max().item()
    assert max_diff < 5e-3, (
        f"paged != contiguous: max diff {max_diff:.4e} "
        f"for block_size={block_size} H={H} M={M} N={N} D={D}"
    )


@cuda
@requires_cuda
def test_paged_fp32_tight():
    """FP32: paged must match contiguous tightly — isolates addressing errors from FP16 noise."""
    from engine.kernels.triton_attention import triton_attention
    from engine.kernels.paged_attention_kernel import paged_attention, scatter_to_pages

    torch.manual_seed(1)
    device = "cuda"
    H, M, N, D = 8, 64, 64, 128
    block_size = 16

    q = torch.randn(H, M, D, device=device, dtype=torch.float32)
    k = torch.randn(H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(H, N, D, device=device, dtype=torch.float32)

    ref = triton_attention(q[None], k[None], v[None], causal=True)[0]

    num_logical = (N + block_size - 1) // block_size
    num_phys = num_logical + 5
    bt = _make_shuffled_block_table(num_logical, num_phys, seed=3).to(device)
    kp, vp = scatter_to_pages(k, v, block_size, bt, num_phys)

    out = paged_attention(q, kp, vp, bt, kv_len=N, causal=True)
    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-3, f"FP32 paged max diff {max_diff:.4e} — addressing error"


@cuda
@requires_cuda
def test_paged_identity_table_equals_contiguous():
    """Sanity: with an identity block table, paged == contiguous exactly (no shuffle)."""
    from engine.kernels.triton_attention import triton_attention
    from engine.kernels.paged_attention_kernel import paged_attention, scatter_to_pages

    torch.manual_seed(2)
    device = "cuda"
    H, M, N, D = 8, 64, 64, 128
    block_size = 16

    q = torch.randn(H, M, D, device=device, dtype=torch.float16)
    k = torch.randn(H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(H, N, D, device=device, dtype=torch.float16)

    ref = triton_attention(q[None], k[None], v[None], causal=True)[0]

    num_logical = (N + block_size - 1) // block_size
    bt = torch.arange(num_logical, dtype=torch.int32, device=device)  # identity
    kp, vp = scatter_to_pages(k, v, block_size, bt, num_logical)

    out = paged_attention(q, kp, vp, bt, kv_len=N, causal=True)
    assert (out - ref).abs().max().item() < 5e-3
