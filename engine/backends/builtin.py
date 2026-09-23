"""The attention backends this engine ships, and what each one needs.

Priorities encode *measured* preference, not novelty. An unmeasured device keeps the
established SDPA gather path as its fallback. A production FlashAttention backend can
take priority once it imports, but the tiled Triton path is an explicit measurement
candidate: it must first pass the PTX, A/B, and token gates on that architecture. Where a
measurement exists it overrides this - see `docs/optimization-journal.md`, and
`policy.py` for the per-device table.

Every `available` returns the reason a backend cannot run here, so `check_hooks` on a new
GPU prints the whole table with reasons instead of failing at the first request.
"""

from __future__ import annotations

import torch

from engine.backends.registry import Backend, Geometry, register


def _triton_available(profile, geometry: Geometry) -> str | None:
    # Geometry and device capability first: those reasons are specific to this model on
    # this GPU and are what an operator needs to read. A missing Triton is environmental
    # and would otherwise mask them everywhere.
    if geometry.head_dim > 128:
        return f"head_dim {geometry.head_dim} exceeds the paged kernels' limit of 128"
    try:
        import triton  # noqa: F401
    except Exception as error:  # pragma: no cover - environment dependent
        return f"triton is not importable ({error})"
    return None


def _flash_available(profile, geometry: Geometry) -> str | None:
    from engine.kernels.flash_paged import supports, unavailable_reason

    reason = unavailable_reason()
    if reason is not None:
        return reason
    dtype = torch.float16 if geometry.dtype == "float16" else torch.bfloat16
    return supports(geometry.block_size, geometry.head_dim, dtype)


def _flash_dense_available(profile, geometry: Geometry) -> str | None:
    from engine.kernels.flash_paged import supports_dense

    if _int8_pool(geometry):
        return "the dense gather path has no INT8 dequantizing variant"
    dtype = torch.float16 if geometry.dtype == "float16" else torch.bfloat16
    return supports_dense(geometry.head_dim, dtype)


def _int8_pool(geometry: Geometry) -> bool:
    return geometry.kv_dtype == "int8"


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _decode_per_head(query, key_pages, value_pages, block_tables, seq_lens, **kwargs):
    from engine.kernels.paged_decode_batched import paged_decode_batched

    kwargs.pop("max_sequence_length", None)
    kwargs.pop("num_splits", None)
    return paged_decode_batched(query, key_pages, value_pages, block_tables, seq_lens, **kwargs)


def _decode_gqa(query, key_pages, value_pages, block_tables, seq_lens, **kwargs):
    from engine.kernels.paged_decode_gqa import paged_decode_gqa

    kwargs.pop("max_sequence_length", None)
    kwargs.pop("num_splits", None)
    return paged_decode_gqa(query, key_pages, value_pages, block_tables, seq_lens, **kwargs)


def _decode_split_k(query, key_pages, value_pages, block_tables, seq_lens, **kwargs):
    from engine.kernels.paged_decode_split_k import paged_decode_split_k

    kwargs.pop("num_splits", None)
    return paged_decode_split_k(query, key_pages, value_pages, block_tables, seq_lens, **kwargs)


def _decode_flash(query, key_pages, value_pages, block_tables, seq_lens, **kwargs):
    from engine.kernels.flash_paged import flash_paged_decode

    kwargs.pop("max_sequence_length", None)
    return flash_paged_decode(query, key_pages, value_pages, block_tables, seq_lens, **kwargs)


def _gqa_available(profile, geometry: Geometry) -> str | None:
    if geometry.gqa_group != 2:
        return f"written for a GQA group of exactly 2, not {geometry.gqa_group}"
    if _int8_pool(geometry):
        return "no INT8 variant"
    return _triton_available(profile, geometry)


def _split_k_available(profile, geometry: Geometry) -> str | None:
    if _int8_pool(geometry):
        return "no INT8 variant"
    return _triton_available(profile, geometry)


register(Backend(
    name="per_head", phase="decode", run=_decode_per_head, available=_triton_available,
    priority=50, summary="Triton, one program per (row, query head); the measured baseline.",
    tags=("triton", "paged", "int8-capable"),
))
register(Backend(
    name="gqa", phase="decode", run=_decode_gqa, available=_gqa_available,
    priority=10,
    summary="Triton, one program per (row, KV head), K/V tile shared by the group. "
            "Measured 0.95-1.06x on the T4: L2 already serves the second read.",
    tags=("triton", "paged", "negative-result"),
))
register(Backend(
    name="split_k", phase="decode", run=_decode_split_k, available=_split_k_available,
    priority=20,
    summary="Triton FlashDecoding: the key range split across programs, then merged. "
            "For small batches and long contexts, where the per-head grid is too narrow.",
    tags=("triton", "paged", "unmeasured"),
))
register(Backend(
    name="flash", phase="decode", run=_decode_flash, available=_flash_available,
    priority=80, graph_safe=False,
    summary="flash_attn_with_kvcache over the pages; splits internally, GQA native.",
    tags=("flash-attn", "paged", "sm80+"),
))


# ---------------------------------------------------------------------------
# Prefill (chunked)
# ---------------------------------------------------------------------------

def _prefill_sdpa(query, key_pages, value_pages, block_tables, start_positions, chunk_lens,
                  *, scale=None, total_len=0, cache=None, **_):
    from engine.kernels.sdpa_prefill import sdpa_paged_prefill

    return sdpa_paged_prefill(
        query, key_pages, value_pages, block_tables, start_positions, chunk_lens,
        scale=scale, total_len=total_len, cache=cache,
    )


def _prefill_per_token(query, key_pages, value_pages, block_tables, start_positions,
                       chunk_lens, *, scale=None, **_):
    from engine.kernels.paged_prefill import paged_prefill

    return paged_prefill(query, key_pages, value_pages, block_tables, start_positions,
                         chunk_lens, scale=scale)


def _prefill_tiled(query, key_pages, value_pages, block_tables, start_positions, chunk_lens,
                   *, scale=None, block_m=None, block_n=None, **_):
    from engine.kernels.tiled_paged_prefill import tiled_paged_prefill

    return tiled_paged_prefill(
        query, key_pages, value_pages, block_tables, start_positions, chunk_lens,
        scale=scale, block_m=block_m, block_n=block_n,
    )


def _prefill_flash(query, key_pages, value_pages, block_tables, start_positions, chunk_lens,
                   *, scale=None, total_len=0, cache=None, **_):
    from engine.kernels.flash_paged import flash_paged_prefill

    return flash_paged_prefill(
        query, key_pages, value_pages, block_tables, start_positions, chunk_lens,
        scale=scale, total_len=total_len, cache=cache,
    )


def _prefill_flash_dense(query, key_pages, value_pages, block_tables, start_positions,
                         chunk_lens, *, scale=None, total_len=0, cache=None, **_):
    from engine.kernels.flash_paged import flash_dense_prefill

    return flash_dense_prefill(
        query, key_pages, value_pages, block_tables, start_positions, chunk_lens,
        scale=scale, total_len=total_len, cache=cache,
    )


def _sdpa_available(profile, geometry: Geometry) -> str | None:
    if _int8_pool(geometry):
        return "the gather path has no INT8 dequantizing variant; use per_token"
    return None


def _tiled_available(profile, geometry: Geometry) -> str | None:
    if profile is not None and profile.sm < 80:
        # Measured, not assumed: on sm_75 `tl.dot` emits FMA rather than mma.sync, and
        # the kernel spills at 255 registers (journal, "never used the tensor cores").
        return f"tl.dot does not reach the tensor cores on sm_{profile.sm}"
    if _int8_pool(geometry):
        return "no INT8 variant"
    return _triton_available(profile, geometry)


register(Backend(
    name="sdpa", phase="prefill", run=_prefill_sdpa, available=_sdpa_available,
    priority=50,
    summary="Gather the prefix pages, then torch SDPA (memory-efficient backend). "
            "The T4 default: -40%/-66% step vs per_token, token-identical to stock.",
    tags=("torch", "gather"),
))
register(Backend(
    name="per_token", phase="prefill", run=_prefill_per_token, available=_triton_available,
    priority=30,
    summary="Triton, one program per (row, head, query token), reads pages in place. "
            "The only chunked path that works with an INT8 pool.",
    tags=("triton", "paged", "int8-capable"),
))
register(Backend(
    name="tiled", phase="prefill", run=_prefill_tiled, available=_tiled_available,
    priority=40,
    summary="Triton FlashAttention structure with tl.dot over paged K/V. Needs sm_80+ "
            "to reach the tensor cores; an explicit candidate gated by "
            "prefill_attention_ab.py --ptx-only and the RTX A/B.",
    tags=("triton", "paged", "sm80+"),
))
register(Backend(
    name="flash", phase="prefill", run=_prefill_flash, available=_flash_available,
    priority=80, graph_safe=False,
    summary="flash_attn_with_kvcache with causal=True over the pages: no gather, no mask.",
    tags=("flash-attn", "paged", "sm80+"),
))
register(Backend(
    name="flash_dense", phase="prefill", run=_prefill_flash_dense,
    available=_flash_dense_available, priority=45, graph_safe=False,
    summary="Gather 16-token pages, then run dense flash_attn_func with native GQA.",
    tags=("flash-attn", "gather", "sm80+"),
))
