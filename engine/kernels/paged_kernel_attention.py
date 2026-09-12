"""K3b: paged-attention kernel wired into live model generation.

This is the culmination of the kernel path. K1 proved the attention math, K2 proved the
block addressing. K3b makes the K2 kernel the model's ACTUAL attention function, reading
K,V directly from the PagedCache's physical block tensors during real generation.

The full paged path:
    1. Qwen3 calls past_key_values.update(k, v, layer_idx)
       -> PagedLayer writes new K,V into its page tensors (M2 storage)
    2. Qwen3 calls the attention interface (this function)
       -> we reach into that layer's key_pages/value_pages and run the K2 kernel on
          them directly. No gather, no contiguous copy. The kernel reads blocks.

Because M2-minimal allocates blocks contiguously (0,1,2,...), the block table is the
identity map. The K2 kernel with an identity table reads pages in logical order — which
is correct. (Non-contiguous shared-pool block tables are M2-full / K4.)

GQA: pages store num_kv_heads; query has num_query_heads. We expand the pages' heads
(repeat_kv) so the kernel sees equal head counts, exactly as M1/M2 did downstream.

Correctness gate (K3b): token-identical greedy generation vs the stock-sdpa reference.
"""

from __future__ import annotations

from typing import Optional

import torch

from engine.kernels.paged_attention_kernel import paged_attention


# ---------------------------------------------------------------------------
# Active-cache handle so the attention fn can find the PagedCache + layer pages.
# The attention interface signature does not receive the cache, so we stash it.
# ---------------------------------------------------------------------------

_ACTIVE_CACHE = None
_KERNEL_CALLS = 0


def set_active_paged_cache(cache) -> None:
    """Register the PagedCache the kernel attention fn should read from."""
    global _ACTIVE_CACHE
    _ACTIVE_CACHE = cache


def clear_active_paged_cache() -> None:
    global _ACTIVE_CACHE
    _ACTIVE_CACHE = None


def kernel_call_count() -> int:
    return _KERNEL_CALLS


def reset_kernel_call_count() -> None:
    global _KERNEL_CALLS
    _KERNEL_CALLS = 0


def _repeat_kv_pages(pages: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand [num_blocks, block_size, kv_heads, D] -> [.., kv_heads*n_rep, D].

    Mirrors repeat_kv along the head axis so the kernel sees query-head count.
    """
    nb, bs, kv_h, d = pages.shape
    if n_rep == 1:
        return pages
    return (
        pages[:, :, :, None, :]
        .expand(nb, bs, kv_h, n_rep, d)
        .reshape(nb, bs, kv_h * n_rep, d)
    )


def paged_kernel_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,       # [B=1, num_q_heads, M, D]
    key: torch.Tensor,         # gathered K from PagedLayer (we IGNORE this)
    value: torch.Tensor,       # gathered V (IGNORED) — we read pages instead
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Attention that runs the K2 paged kernel on PagedCache blocks directly.

    We deliberately ignore the `key`/`value` arguments (HF's gathered contiguous K,V)
    and instead read the physical page tensors the PagedLayer just wrote to. This is
    what makes it a REAL paged-attention path: the kernel reads blocks, not a copy.
    """
    global _KERNEL_CALLS
    _KERNEL_CALLS += 1

    assert _ACTIVE_CACHE is not None, "call set_active_paged_cache(cache) before generating"
    layer_idx = module.layer_idx
    paged_layer = _ACTIVE_CACHE._paged_layers[layer_idx]

    key_pages = paged_layer.key_pages       # [num_blocks, block_size, kv_heads, D]
    value_pages = paged_layer.value_pages
    seq_len = paged_layer.seq_len
    block_size = paged_layer.block_size_tokens

    B, num_q_heads, M, D = query.shape
    assert B == 1, "K3b handles batch=1 (single sequence)"
    kv_heads = key_pages.shape[2]
    n_rep = num_q_heads // kv_heads

    # Expand pages to query-head count (GQA)
    kp = _repeat_kv_pages(key_pages, n_rep)     # [nb, bs, num_q_heads, D]
    vp = _repeat_kv_pages(value_pages, n_rep)

    # Identity block table: logical block i -> physical block i (M2 contiguous alloc)
    num_logical_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(num_logical_blocks, device=query.device, dtype=torch.int32)

    # Query: drop batch dim -> [num_q_heads, M, D]
    q = query[0]

    # Causal: during prefill M == seq_len (causal). During decode M == 1, the single
    # query attends to all seq_len keys (position seq_len-1), so effectively non-causal
    # for that one row. The kernel's causal mask with offs_m offset handles prefill;
    # for decode we must map the single query's position to seq_len-1.
    # Simplest correct handling: prefill uses causal over the square; decode (M==1)
    # attends to all keys (causal=False, since the one new token sees the whole cache).
    causal = (M == seq_len) and (M > 1)

    out = paged_attention(
        q, kp, vp, block_table, kv_len=seq_len, scale=scaling, causal=causal,
    )
    # out: [num_q_heads, M, D] -> HF expects [B, M, H, D] (transposed form)
    out = out.unsqueeze(0).transpose(1, 2).contiguous()   # [B, M, num_q_heads, D]
    return out, None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PAGED_KERNEL_ATTENTION_NAME = "paged_kernel"


def register_paged_kernel_attention() -> str:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[PAGED_KERNEL_ATTENTION_NAME] = paged_kernel_attention_forward
    return PAGED_KERNEL_ATTENTION_NAME


def enable_paged_kernel_attention(model) -> None:
    """Register the kernel attention fn and select it on the model."""
    register_paged_kernel_attention()
    model.config._attn_implementation = PAGED_KERNEL_ATTENTION_NAME
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = PAGED_KERNEL_ATTENTION_NAME
