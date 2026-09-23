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
@pytest.mark.parametrize("block_n", [64, 128])
@pytest.mark.parametrize("num_q_heads,kv_heads,D", [
    (8, 8, 128),    # no GQA
    (16, 8, 128),   # GQA 2:1 (Qwen3-0.6B)
    (8, 2, 64),     # GQA 4:1, smaller dim
])
def test_batched_decode_matches_per_sequence(
    block_size, block_n, num_q_heads, kv_heads, D,
):
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
                               block_tables, seq_lens_t, block_n=block_n)

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


def test_t4_regime_selector_boundaries() -> None:
    from engine.kernels.paged_decode_config import select_paged_decode_config

    assert select_paged_decode_config(1, 1) == (64, 4)
    assert select_paged_decode_config(127, 16) == (64, 4)
    assert select_paged_decode_config(128, 1) == (128, 4)
    assert select_paged_decode_config(2048, 16) == (128, 4)
    assert select_paged_decode_config(64, 8, "gqa") == (64, 4)
    assert select_paged_decode_config(2048, 16, "gqa") == (128, 4)
    with pytest.raises(ValueError):
        select_paged_decode_config(0, 1)


@cuda
@requires_cuda
def test_length_offset_matches_materialized_lengths():
    """Kernel-side +1 must equal passing an allocated incremented length tensor."""
    from engine.kernels.paged_decode_batched import paged_decode_batched

    torch.manual_seed(19)
    device, dtype = "cuda", torch.float16
    num_heads, D, block_size = 8, 128, 16
    lengths = [7, 16, 33]
    queries = [
        torch.randn(num_heads, 1, D, device=device, dtype=dtype) for _ in lengths
    ]
    seqs_kv = [
        (
            torch.randn(num_heads, n, D, device=device, dtype=dtype),
            torch.randn(num_heads, n, D, device=device, dtype=dtype),
        )
        for n in lengths
    ]
    key_pages, value_pages, block_tables, seq_lens = _build_shared_pool(
        seqs_kv, block_size, num_heads, D, device, dtype, seed=23,
    )
    query = torch.stack(queries)

    materialized = paged_decode_batched(
        query, key_pages, value_pages, block_tables, seq_lens, block_n=64,
    )
    kernel_offset = paged_decode_batched(
        query,
        key_pages,
        value_pages,
        block_tables,
        seq_lens - 1,
        block_n=64,
        length_offset=1,
    )
    assert torch.equal(kernel_offset, materialized)


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


@cuda
@requires_cuda
@pytest.mark.parametrize("block_n", [32, 64, 128])
@pytest.mark.parametrize("num_q_heads,kv_heads,D", [
    (16, 8, 128),   # GQA 2:1 (Qwen3-0.6B)
    (4, 2, 64),     # GQA 2:1, smaller dim
])
def test_gqa_shared_decode_matches_per_head_kernel(block_n, num_q_heads, kv_heads, D):
    """One program per KV head, all of its query heads: bit-close to the per-head kernel
    and to SDPA, with an offset length and shuffled block tables."""
    from engine.kernels.paged_decode_batched import paged_decode_batched
    from engine.kernels.paged_decode_gqa import paged_decode_gqa

    torch.manual_seed(1)
    device, dtype, block_size = "cuda", torch.float16, 16
    seq_lens = [1, 7, 16, 33, 64, 100, 257]
    queries, seqs_kv = [], []
    for n in seq_lens:
        queries.append(torch.randn(num_q_heads, 1, D, device=device, dtype=dtype))
        seqs_kv.append((torch.randn(kv_heads, n, D, device=device, dtype=dtype),
                        torch.randn(kv_heads, n, D, device=device, dtype=dtype)))
    key_pages, value_pages, block_tables, seq_lens_t = _build_shared_pool(
        seqs_kv, block_size, kv_heads, D, device, dtype,
    )
    query_batched = torch.stack(queries, dim=0)
    per_head = paged_decode_batched(query_batched, key_pages, value_pages, block_tables, seq_lens_t)
    shared = paged_decode_gqa(query_batched, key_pages, value_pages, block_tables, seq_lens_t,
                              block_n=block_n)
    torch.testing.assert_close(shared, per_head, rtol=2e-3, atol=2e-3)

    n_rep = num_q_heads // kv_heads
    for s, n in enumerate(seq_lens):
        k = seqs_kv[s][0].repeat_interleave(n_rep, dim=0)
        v = seqs_kv[s][1].repeat_interleave(n_rep, dim=0)
        ref = _sdpa_decode_reference(queries[s], k, v)
        assert (shared[s] - ref).abs().max().item() < 5e-3, f"sequence {s} (len {n})"

    # The engine stores pre-write lengths and asks the kernel to add one.
    shorter = seq_lens_t - 1
    offset = paged_decode_gqa(query_batched, key_pages, value_pages, block_tables, shorter,
                              block_n=block_n, length_offset=1)
    torch.testing.assert_close(offset, shared, rtol=2e-3, atol=2e-3)


@cuda
@requires_cuda
@pytest.mark.parametrize("splits", [2, 4, 8])
@pytest.mark.parametrize("block_n", [32, 64])
def test_split_k_decode_matches_the_single_pass_kernel(splits, block_n):
    """Splitting the key range and merging the partial softmaxes changes nothing."""
    from engine.kernels.paged_decode_batched import paged_decode_batched
    from engine.kernels.paged_decode_split_k import paged_decode_split_k

    torch.manual_seed(5)
    device, dtype, block_size = "cuda", torch.float16, 16
    num_q_heads, kv_heads, D = 16, 8, 128
    # Lengths that straddle split boundaries, including a row with a single key and one
    # long enough that late splits are empty.
    seq_lens = [1, 15, 16, 17, 63, 200, 511]
    queries, seqs_kv = [], []
    for n in seq_lens:
        queries.append(torch.randn(num_q_heads, 1, D, device=device, dtype=dtype))
        seqs_kv.append((torch.randn(kv_heads, n, D, device=device, dtype=dtype),
                        torch.randn(kv_heads, n, D, device=device, dtype=dtype)))
    key_pages, value_pages, block_tables, seq_lens_t = _build_shared_pool(
        seqs_kv, block_size, kv_heads, D, device, dtype,
    )
    query = torch.stack(queries, dim=0)
    reference = paged_decode_batched(query, key_pages, value_pages, block_tables, seq_lens_t,
                                     block_n=block_n)
    actual = paged_decode_split_k(
        query, key_pages, value_pages, block_tables, seq_lens_t, block_n=block_n,
        splits=splits, max_sequence_length=max(seq_lens),
    )
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)

    # And against SDPA directly, so a shared bug in both paged kernels cannot hide.
    n_rep = num_q_heads // kv_heads
    for s, n in enumerate(seq_lens):
        k = seqs_kv[s][0].repeat_interleave(n_rep, dim=0)
        v = seqs_kv[s][1].repeat_interleave(n_rep, dim=0)
        expected = _sdpa_decode_reference(queries[s], k, v)
        assert (actual[s] - expected).abs().max().item() < 5e-3, f"sequence {s} (len {n})"


@cuda
@requires_cuda
def test_split_k_honours_the_length_offset_like_the_single_pass_kernel():
    from engine.kernels.paged_decode_split_k import paged_decode_split_k

    torch.manual_seed(6)
    device, dtype, block_size = "cuda", torch.float16, 16
    seqs_kv = [(torch.randn(8, n, 128, device=device, dtype=dtype),
                torch.randn(8, n, 128, device=device, dtype=dtype)) for n in (33, 129)]
    key_pages, value_pages, block_tables, seq_lens_t = _build_shared_pool(
        seqs_kv, block_size, 8, 128, device, dtype,
    )
    query = torch.randn(2, 16, 1, 128, device=device, dtype=dtype)
    full = paged_decode_split_k(query, key_pages, value_pages, block_tables, seq_lens_t,
                                splits=4, max_sequence_length=129)
    offset = paged_decode_split_k(query, key_pages, value_pages, block_tables, seq_lens_t - 1,
                                  splits=4, length_offset=1, max_sequence_length=129)
    torch.testing.assert_close(offset, full, rtol=2e-3, atol=2e-3)


def test_choose_splits_widens_a_narrow_grid_and_leaves_a_wide_one_alone() -> None:
    # Imported from the policy module, which has no Triton dependency, so the heuristic
    # that decides whether to split is testable wherever the tests run.
    from engine.kernels.paged_decode_config import choose_splits

    # One request, 16 heads, long context: 16 programs on 40 SMs wants splitting.
    assert choose_splits(1, 16, 2048, multiprocessors=40) >= 4
    # Sixteen requests at 16 heads is already 256 programs; splitting only adds a merge.
    assert choose_splits(16, 16, 2048, multiprocessors=40) == 1
    # A short context cannot be split into useful tiles whatever the grid says.
    assert choose_splits(1, 1, 64, multiprocessors=108, block_n=64) == 1
    for splits in (choose_splits(rows, 16, 4096) for rows in (1, 2, 4, 8)):
        assert splits & (splits - 1) == 0, "splits must be a power of two for tl.arange"
