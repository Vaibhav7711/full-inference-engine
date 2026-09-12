"""K4 correctness: batched multi-sequence decode == per-sequence single decode.

The gate: run S sequences of DIFFERENT lengths with DIFFERENT (shuffled) block tables
through the batched K4 kernel in ONE launch. Then run each sequence separately through
a single-sequence reference. Assert every sequence's output matches.

Because the single-sequence path is verified (K2 == K1 == SDPA), transitively K4 == SDPA
per sequence. And because sequences have different lengths and shuffled block tables in a
shared pool, a bug in per-sequence addressing, length handling, or the shared-pool layout
would produce wrong output.

The reference here is a direct PyTorch SDPA per sequence (simplest trusted oracle for the
decode case: one query attends to all of that sequence's keys).

Run:
    python -m pytest tests/kernels/test_paged_decode_batched.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _sdpa_decode_reference(q_1, k_full, v_full):
    """Single-sequence decode reference: q [H,1,D] attends to k,v [H,N,D] (all keys)."""
    # scaled dot-product, non-causal (the one query sees all N keys)
    return torch.nn.functional.scaled_dot_product_attention(
        q_1[None], k_full[None], v_full[None], is_causal=False,
    )[0]  # [H, 1, D]


def _build_shared_pool(seqs_kv, block_size, kv_heads, D, device, dtype, seed=7):
    """Scatter multiple sequences' K,V into ONE shared pool with shuffled block tables.

    seqs_kv: list of (k [kv_heads, N, D], v [kv_heads, N, D]) per sequence.
    Returns key_pages, value_pages, block_tables [S, max_blocks], seq_lens [S].
    """
    S = len(seqs_kv)
    seq_lens = [k.shape[1] for k, v in seqs_kv]
    blocks_per_seq = [(n + block_size - 1) // block_size for n in seq_lens]
    total_blocks = sum(blocks_per_seq) + 5  # extra unused blocks
    max_blocks = max(blocks_per_seq)

    # Assign physical blocks to sequences from a shuffled pool
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_blocks, generator=g).tolist()

    key_pages = torch.zeros((total_blocks, block_size, kv_heads, D), device=device, dtype=dtype)
    value_pages = torch.zeros_like(key_pages)
    block_tables = torch.zeros((S, max_blocks), dtype=torch.int32, device=device)

    cursor = 0
    for s, (k, v) in enumerate(seqs_kv):
        n = seq_lens[s]
        nb = blocks_per_seq[s]
        phys_blocks = perm[cursor:cursor + nb]
        cursor += nb
        for lb in range(nb):
            block_tables[s, lb] = phys_blocks[lb]
        # scatter tokens
        for pos in range(n):
            lb = pos // block_size
            off = pos % block_size
            phys = phys_blocks[lb]
            key_pages[phys, off] = k[:, pos, :]     # [kv_heads, D]
            value_pages[phys, off] = v[:, pos, :]

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return key_pages, value_pages, block_tables, seq_lens_t


@cuda
@requires_cuda
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_q_heads,kv_heads,D", [
    (8, 8, 128),    # no GQA
    (16, 8, 128),   # GQA 2:1 (Qwen3-0.6B)
    (8, 2, 64),     # GQA 4:1, smaller dim
])
def test_batched_decode_matches_per_sequence(block_size, num_q_heads, kv_heads, D):
    from engine.kernels.paged_decode_batched import paged_decode_batched

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # S sequences of DIFFERENT lengths
    seq_lens = [7, 16, 33, 64, 100]
    S = len(seq_lens)

    # Per-sequence Q (1 token) and K,V (full history)
    queries = []
    seqs_kv = []
    for n in seq_lens:
        q = torch.randn(num_q_heads, 1, D, device=device, dtype=dtype)
        k = torch.randn(kv_heads, n, D, device=device, dtype=dtype)
        v = torch.randn(kv_heads, n, D, device=device, dtype=dtype)
        queries.append(q)
        seqs_kv.append((k, v))

    # Build shared pool with shuffled block tables
    key_pages, value_pages, block_tables, seq_lens_t = _build_shared_pool(
        seqs_kv, block_size, kv_heads, D, device, dtype,
    )

    # Batched query tensor [S, H, 1, D]
    query_batched = torch.stack(queries, dim=0)   # [S, H, 1, D]

    # --- K4 batched: one launch ---
    out = paged_decode_batched(query_batched, key_pages, value_pages,
                               block_tables, seq_lens_t, block_n=64)

    # --- Reference: each sequence separately via SDPA (with GQA expansion) ---
    def repeat_kv(x, n_rep):
        h, s, d = x.shape
        if n_rep == 1:
            return x
        return x[:, None, :, :].expand(h, n_rep, s, d).reshape(h * n_rep, s, d)

    n_rep = num_q_heads // kv_heads
    for s in range(S):
        k_exp = repeat_kv(seqs_kv[s][0], n_rep)   # [num_q_heads, N, D]
        v_exp = repeat_kv(seqs_kv[s][1], n_rep)
        ref = _sdpa_decode_reference(queries[s], k_exp, v_exp)   # [H, 1, D]
        got = out[s]                                              # [H, 1, D]
        max_diff = (got - ref).abs().max().item()
        assert max_diff < 5e-3, (
            f"sequence {s} (len {seq_lens[s]}) batched != reference: "
            f"max diff {max_diff:.4e}  block_size={block_size} "
            f"heads={num_q_heads}/{kv_heads} D={D}"
        )


@cuda
@requires_cuda
def test_batched_decode_fp32_tight():
    """FP32: batched == per-sequence tightly, isolating addressing/length errors."""
    from engine.kernels.paged_decode_batched import paged_decode_batched

    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.float32
    num_q_heads, kv_heads, D, block_size = 8, 8, 128, 16
    seq_lens = [10, 25, 50]
    S = len(seq_lens)

    queries, seqs_kv = [], []
    for n in seq_lens:
        queries.append(torch.randn(num_q_heads, 1, D, device=device, dtype=dtype))
        seqs_kv.append((
            torch.randn(kv_heads, n, D, device=device, dtype=dtype),
            torch.randn(kv_heads, n, D, device=device, dtype=dtype),
        ))

    key_pages, value_pages, block_tables, seq_lens_t = _build_shared_pool(
        seqs_kv, block_size, kv_heads, D, device, dtype, seed=11,
    )
    query_batched = torch.stack(queries, dim=0)

    out = paged_decode_batched(query_batched, key_pages, value_pages,
                               block_tables, seq_lens_t, block_n=32)

    for s in range(S):
        ref = _sdpa_decode_reference(queries[s], seqs_kv[s][0], seqs_kv[s][1])
        max_diff = (out[s] - ref).abs().max().item()
        assert max_diff < 1e-3, (
            f"FP32 sequence {s} max diff {max_diff:.4e} — addressing/length error"
        )


@cuda
@requires_cuda
def test_batched_equals_separate_launches():
    """The core claim: ONE batched launch == S separate single-sequence launches.

    Verifies K4 (batched) against K2 (single-sequence) directly — the exact win
    continuous batching relies on.
    """
    from engine.kernels.paged_decode_batched import paged_decode_batched
    from engine.kernels.paged_attention_kernel import paged_attention

    torch.manual_seed(2)
    device = "cuda"
    dtype = torch.float16
    num_heads, D, block_size = 8, 128, 16   # no GQA for direct K2 comparison
    seq_lens = [8, 20, 40]
    S = len(seq_lens)

    queries, seqs_kv = [], []
    for n in seq_lens:
        queries.append(torch.randn(num_heads, 1, D, device=device, dtype=dtype))
        seqs_kv.append((
            torch.randn(num_heads, n, D, device=device, dtype=dtype),
            torch.randn(num_heads, n, D, device=device, dtype=dtype),
        ))

    key_pages, value_pages, block_tables, seq_lens_t = _build_shared_pool(
        seqs_kv, block_size, num_heads, D, device, dtype, seed=13,
    )
    query_batched = torch.stack(queries, dim=0)

    # K4 batched
    out_batched = paged_decode_batched(query_batched, key_pages, value_pages,
                                       block_tables, seq_lens_t, block_n=64)

    # K2 per sequence, using each sequence's own block table row
    for s in range(S):
        n = seq_lens[s]
        nb = (n + block_size - 1) // block_size
        bt_s = block_tables[s, :nb].contiguous()
        out_k2 = paged_attention(
            queries[s], key_pages, value_pages, bt_s, kv_len=n,
            causal=False,  # decode: single query sees all keys
        )   # [H, 1, D]
        max_diff = (out_batched[s] - out_k2).abs().max().item()
        assert max_diff < 5e-3, (
            f"sequence {s}: K4 batched != K2 single launch, max diff {max_diff:.4e}"
        )
