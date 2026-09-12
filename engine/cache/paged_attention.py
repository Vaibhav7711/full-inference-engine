"""M1: paged-attention read path.

This is the first stage where our block-structured KV addressing participates in the
*real* model attention call, not just a standalone gather test.

Strategy (M1 / Tier 1 — "paged read path"):
    - Register a custom attention function in Transformers' attention registry.
    - Inside it, before computing scaled dot-product attention, route the incoming
      K and V tensors through a block-structured round trip:
          contiguous K,V  ->  scatter into physical pages  ->  gather back via block table
    - Assert the gathered K,V are bit-identical to the input K,V. This proves the
      block addressing and gather are correct *inside the attention path itself*.
    - Run SDPA on the gathered tensors and return.

Honest scope of M1:
    HuggingFace's DynamicCache still owns the authoritative K,V storage. We gather in
    parallel and verify equivalence. This proves the addressing is correct in the real
    attention call. It does NOT yet mean our block store is the sole physical home of
    the cache — that is M2 (paged storage path), where a custom cache class replaces
    DynamicCache as the authoritative store.

Tensor contract (verified for Qwen3-0.6B, transformers 5.16.1, sdpa):
    query : [batch, num_query_heads, q_len, head_dim]
    key   : [batch, num_kv_heads,   kv_len, head_dim]   (GQA: kv_heads < query_heads)
    value : [batch, num_kv_heads,   kv_len, head_dim]
    The GQA repeat_kv to num_query_heads happens *after* the cache read, inside the
    attention function (repeat_kv / enable_gqa). Our blocks therefore store num_kv_heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .block_table import BlockTable


# ---------------------------------------------------------------------------
# Block round-trip for a single [batch, heads, seq, head_dim] tensor
# ---------------------------------------------------------------------------

def _scatter_into_pages(
    kv: torch.Tensor,
    block_size_tokens: int,
    physical_block_ids: tuple[int, ...],
    num_physical_blocks: int,
) -> torch.Tensor:
    """Scatter a contiguous [B, H, S, D] tensor into physical pages.

    Pages are shaped [num_physical_blocks, B, H, block_size_tokens, D]. Logical token i
    of the sequence lands in physical block physical_block_ids[i // block_size] at
    offset i % block_size. Unused page slots stay zero.

    This mimics how a real paged store lays K,V out in non-contiguous physical blocks.
    """
    batch, heads, seq_len, head_dim = kv.shape
    pages = kv.new_zeros((num_physical_blocks, batch, heads, block_size_tokens, head_dim))
    for logical_tok in range(seq_len):
        logical_block, offset = divmod(logical_tok, block_size_tokens)
        physical_block = physical_block_ids[logical_block]
        pages[physical_block, :, :, offset, :] = kv[:, :, logical_tok, :]
    return pages


def _gather_from_pages(
    pages: torch.Tensor,
    block_table: BlockTable,
    sequence_length: int,
) -> torch.Tensor:
    """Gather logical [B, H, S, D] order back from physical pages using the block table.

    Inverse of _scatter_into_pages. Selects the physical blocks named by the table,
    concatenates them in logical order, and slices to sequence_length.
    """
    device = pages.device
    indices = torch.tensor(block_table.physical_block_ids, device=device, dtype=torch.long)
    # pages: [num_blocks, B, H, block_size, D] -> select logical order
    selected = pages.index_select(0, indices)  # [num_logical_blocks, B, H, block_size, D]
    num_logical_blocks, batch, heads, block_size, head_dim = selected.shape
    # Merge (logical_block, block_token) -> flat token axis, then move to [B, H, S, D]
    merged = selected.permute(1, 2, 0, 3, 4).reshape(batch, heads, num_logical_blocks * block_size, head_dim)
    return merged[:, :, :sequence_length, :]


def block_round_trip(
    kv: torch.Tensor,
    block_size_tokens: int,
) -> torch.Tensor:
    """Full scatter->gather round trip for one K or V tensor.

    Allocates a fresh block table covering the sequence, scatters into pages, gathers
    back. The result must equal the input exactly. This is the operation we insert into
    the attention path to prove addressing correctness.
    """
    batch, heads, seq_len, head_dim = kv.shape
    num_blocks_needed = (seq_len + block_size_tokens - 1) // block_size_tokens
    # Use a non-identity physical layout so the block table is actually exercised:
    # reverse the block order, so logical block k -> physical block (n-1-k).
    physical_block_ids = tuple(range(num_blocks_needed - 1, -1, -1))
    table = BlockTable(
        request_id="round_trip",
        block_size_tokens=block_size_tokens,
        physical_block_ids=physical_block_ids,
    )
    pages = _scatter_into_pages(kv, block_size_tokens, physical_block_ids, num_blocks_needed)
    gathered = _gather_from_pages(pages, table, seq_len)
    return gathered


# ---------------------------------------------------------------------------
# Custom attention function
# ---------------------------------------------------------------------------

@dataclass
class PagedAttentionConfig:
    """Runtime configuration + verification counters for the paged attention path."""
    block_size_tokens: int = 16
    verify: bool = True          # assert gather == input every call (M1 correctness)
    atol: float = 0.0            # exact match required (same dtype round trip)
    # Counters (filled during forward passes)
    calls: int = 0
    max_abs_diff: float = 0.0


# Module-level config so the registered function can see it without HF passing it through.
_PAGED_CONFIG = PagedAttentionConfig()


def get_paged_config() -> PagedAttentionConfig:
    return _PAGED_CONFIG


def reset_paged_config(block_size_tokens: int = 16, verify: bool = True) -> PagedAttentionConfig:
    global _PAGED_CONFIG
    _PAGED_CONFIG = PagedAttentionConfig(block_size_tokens=block_size_tokens, verify=verify)
    return _PAGED_CONFIG


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA expansion: [B, kv_heads, S, D] -> [B, kv_heads*n_rep, S, D].

    Mirrors transformers' repeat_kv. Needed when we compute attention manually rather
    than relying on SDPA's enable_gqa.
    """
    batch, kv_heads, seq_len, head_dim = hidden.shape
    if n_rep == 1:
        return hidden
    hidden = hidden[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
    return hidden.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


def paged_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Paged read-path attention.

    Signature matches transformers' eager/sdpa attention functions so it can be
    registered and selected via config._attn_implementation.

    K and V arrive contiguous (from DynamicCache). We round-trip them through the block
    structure, verify equality, then run standard SDPA on the gathered tensors.
    """
    cfg = _PAGED_CONFIG
    cfg.calls += 1

    # --- Block round trip on K and V ---
    gathered_key = block_round_trip(key, cfg.block_size_tokens)
    gathered_value = block_round_trip(value, cfg.block_size_tokens)

    # --- Verify addressing correctness inside the real attention call ---
    if cfg.verify:
        k_diff = (gathered_key - key).abs().max().item()
        v_diff = (gathered_value - value).abs().max().item()
        cfg.max_abs_diff = max(cfg.max_abs_diff, k_diff, v_diff)
        if k_diff > cfg.atol or v_diff > cfg.atol:
            raise AssertionError(
                f"paged gather mismatch: k_diff={k_diff}, v_diff={v_diff} "
                f"(block_size={cfg.block_size_tokens})"
            )

    # --- GQA expansion (query heads > kv heads) ---
    num_query_heads = query.shape[1]
    num_kv_heads = gathered_key.shape[1]
    n_rep = num_query_heads // num_kv_heads
    key_expanded = _repeat_kv(gathered_key, n_rep)
    value_expanded = _repeat_kv(gathered_value, n_rep)

    # --- Standard SDPA on gathered tensors ---
    # attention_mask from HF is already the additive/boolean mask for this step.
    is_causal = attention_mask is None and query.shape[2] > 1

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key_expanded,
        value_expanded,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
        is_causal=is_causal,
    )
    # transformers expects [B, S, H, D] (transpose back happens in the caller)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PAGED_ATTENTION_NAME = "paged_read_path"


def register_paged_attention() -> str:
    """Register the paged attention function in Transformers' global registry.

    Returns the name to assign to model.config._attn_implementation.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[PAGED_ATTENTION_NAME] = paged_sdpa_attention_forward
    return PAGED_ATTENTION_NAME


def enable_paged_attention_on_model(model, block_size_tokens: int = 16, verify: bool = True) -> None:
    """Register the paged attention fn and switch the model to use it.

    After this call, every attention layer routes K,V through the block round trip.
    """
    register_paged_attention()
    reset_paged_config(block_size_tokens=block_size_tokens, verify=verify)
    model.config._attn_implementation = PAGED_ATTENTION_NAME
    # Some models cache the resolved attention function per-layer; set on config is
    # the supported path in transformers 5.x. Force config propagation:
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = PAGED_ATTENTION_NAME
