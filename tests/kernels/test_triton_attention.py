"""K1 correctness: Triton attention kernel must match torch SDPA.

This is the K1 gate. It proves the attention MATH (tiling, online softmax, causal
masking) is correct in isolation, BEFORE paged addressing is added in K2. If a later
paged stage produces wrong output, we know the bug is in addressing, not the math,
because this test already locked the math down.

Run:
    python -m pytest tests/kernels/test_triton_attention.py -v
    python -m pytest tests/kernels/test_triton_attention.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _sdpa_reference(q, k, v, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal,
    )


@cuda
@requires_cuda
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("B,H,M,N,D", [
    (1, 1, 64, 64, 64),      # square, single head
    (1, 8, 128, 128, 128),   # Qwen3-like head dim, 8 heads
    (2, 4, 100, 100, 64),    # batch 2, non-power-of-2 length
    (1, 16, 37, 37, 128),    # odd length, many heads
    (1, 8, 1, 200, 128),     # decode: single query, long KV (causal covers all)
    (1, 8, 200, 200, 128),   # prefill-length
    (1, 4, 256, 256, 64),    # larger, crosses multiple tiles
])
def test_triton_attention_matches_sdpa(B, H, M, N, D, causal):
    from engine.kernels.triton_attention import triton_attention

    torch.manual_seed(0)
    device = "cuda"
    # For causal with M != N, SDPA's is_causal assumes aligned square; only test
    # causal when M == N (the standard self-attention case), or M==1 decode.
    if causal and M != N and M != 1:
        pytest.skip("causal reference only well-defined for M==N or single-query decode")

    q = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

    # For M==1 decode with causal, the single query attends to ALL keys (position N-1),
    # so causal is effectively non-causal for that one query. Use causal=False reference.
    ref_causal = causal and (M == N)
    ref = _sdpa_reference(q, k, v, ref_causal)
    out = triton_attention(q, k, v, causal=(causal and M == N), block_m=64, block_n=64)

    # FP16 attention: allow modest tolerance (accumulation differences)
    max_diff = (out - ref).abs().max().item()
    mean_diff = (out - ref).abs().mean().item()
    assert max_diff < 2e-2, (
        f"max diff {max_diff:.4e} (mean {mean_diff:.4e}) exceeds tolerance "
        f"for B={B} H={H} M={M} N={N} D={D} causal={causal}"
    )


@cuda
@requires_cuda
def test_triton_attention_fp32_tight():
    """In FP32 the match should be very tight — isolates any math error from FP16 noise."""
    from engine.kernels.triton_attention import triton_attention

    torch.manual_seed(1)
    device = "cuda"
    q = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float32)
    k = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float32)
    v = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float32)

    ref = _sdpa_reference(q, k, v, causal=True)
    out = triton_attention(q, k, v, causal=True, block_m=64, block_n=64)

    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-3, f"FP32 max diff {max_diff:.4e} too large — math error"


@cuda
@requires_cuda
def test_triton_attention_gqa_expanded():
    """GQA case: caller expands KV heads to match Q heads, kernel treats them equally."""
    from engine.kernels.triton_attention import triton_attention

    torch.manual_seed(2)
    device = "cuda"
    B, num_q_heads, num_kv_heads, M, D = 1, 16, 8, 64, 128
    n_rep = num_q_heads // num_kv_heads

    q = torch.randn(B, num_q_heads, M, D, device=device, dtype=torch.float16)
    k_small = torch.randn(B, num_kv_heads, M, D, device=device, dtype=torch.float16)
    v_small = torch.randn(B, num_kv_heads, M, D, device=device, dtype=torch.float16)

    # Expand KV heads (repeat_kv)
    def repeat_kv(x, n):
        b, h, s, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)

    k = repeat_kv(k_small, n_rep)
    v = repeat_kv(v_small, n_rep)

    ref = _sdpa_reference(q, k, v, causal=True)
    out = triton_attention(q, k, v, causal=True)

    max_diff = (out - ref).abs().max().item()
    assert max_diff < 2e-2, f"GQA max diff {max_diff:.4e} too large"


@cuda
@requires_cuda
@pytest.mark.parametrize("block_m,block_n", [(32, 32), (64, 64), (32, 64), (128, 64)])
def test_triton_attention_block_sizes(block_m, block_n):
    """Different tile sizes must all produce the same correct result."""
    from engine.kernels.triton_attention import triton_attention

    torch.manual_seed(3)
    device = "cuda"
    q = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 128, 128, device=device, dtype=torch.float16)

    ref = _sdpa_reference(q, k, v, causal=True)
    out = triton_attention(q, k, v, causal=True, block_m=block_m, block_n=block_n)

    max_diff = (out - ref).abs().max().item()
    assert max_diff < 2e-2, f"block ({block_m},{block_n}) max diff {max_diff:.4e}"
