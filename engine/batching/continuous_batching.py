"""Real continuous batching: full batched decode forward over paged KV (D1-D3).

This is the system the whole kernel path was building toward. It runs the FULL
transformer batched over N sequences of different lengths, each reading its own paged KV
blocks, driven by a scheduler. The K4 kernel handles the one hard part (attention over
per-sequence KV); HuggingFace batches everything else (projections, MLP, norms) naturally
because those are per-token operations.

Verified design facts (Qwen3-0.6B, transformers 5.16.1):
    - model.forward accepts per-row position_ids for a [N,1] batched decode.
    - RoPE is applied inside Qwen3Attention BEFORE the attention function, from
      position_embeddings computed from position_ids. So we pass per-sequence positions;
      the model rotates Q,K per row; our attention fn receives rotated tensors.
    - Historical K in the pool is already rotated (stored post-RoPE during prefill/decode),
      so everything stays consistently in rotated space.

The batched decode step (the crux):
    1. Stack N sequences' one new token -> input_ids [N, 1].
    2. Per-sequence position_ids [N, 1] = each sequence's current length.
    3. Run model([N,1]) with our K4 attention function registered.
    4. Per layer, the attention fn:
         a. WRITES each sequence's new (rotated) K,V into the pool at its next slot.
         b. Runs K4 batched decode: each query attends to its sequence's pool blocks.
    5. After all layers, each sequence's length grows by 1.
    6. Sample next token per sequence from logits [N, 1, vocab].

Staged tests (run in order): test_d1 (prefill), test_d2 (one decode step), test_d3 (loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import torch

from engine.cache import KVBlockManager, PrefixCache
from engine.kernels.kv_write import write_decode_kv
from engine.kernels.paged_decode_batched import (
    paged_decode_batched,
)
from engine.kernels.paged_decode_config import select_paged_decode_config

# Attention implementations live in `engine/backends`, which knows what each one needs
# and which of them can run on this device. `prefill_attention` / `decode_attention`
# accept any registered name or "auto"; these tuples remain for callers that enumerate
# the built-ins (benchmark arms, older scripts).
PREFILL_ATTENTION_KINDS = ("per_token", "sdpa", "tiled", "flash")
DECODE_ATTENTION_KINDS = ("per_head", "gqa", "split_k", "flash")


def _prefill_context_bucket(total_len: int, floor: int = 256, capacity: int | None = None) -> int:
    """Gathered-prefix length a captured SDPA prefill graph is shaped for.

    Powers of two from ``floor``, clamped to the pool's token capacity: the gather cannot
    produce more tokens than the pool holds. Unclamped, a 1536-block (24,576-token) pool
    rounded up to a 32,768-token bucket, whose capture failed and - because one failed
    capture disables graphs for the kind - left every SDPA prefill and fused step eager.
    """
    bucket = floor
    while bucket < total_len:
        bucket *= 2
    if capacity is not None:
        bucket = min(bucket, capacity)
    return bucket
from engine.backends import Geometry, defaults_for, resolve
from engine.batching.sampler import BatchedSampler
from engine.runtime import GenerationRequest, RequestState
from engine.runtime.sampling import GREEDY, SamplingParams
from engine.scheduler import FCFSScheduler


# ---------------------------------------------------------------------------
# Batched decode context — stashed so the attention fn can reach per-seq metadata
# ---------------------------------------------------------------------------

@dataclass
class _BatchContext:
    """Everything the K4 attention fn needs for one batched decode step."""
    key_pool: list          # per-layer [num_blocks, block_size, kv_heads, D]
    value_pool: list
    block_tables: torch.Tensor   # [N, max_blocks] int32, per-sequence
    seq_lens: torch.Tensor       # [N] int32, KV length BEFORE this step's new token
    block_size: int
    decode_block_n: int
    decode_num_warps: int
    decode_num_splits: int = 0
    key_scale_pool: list | None = None
    value_scale_pool: list | None = None
    # "per_head": one program per (row, query head). "gqa": one per (row, KV head), the
    # K/V tile read once for the whole group. See `paged_decode_gqa`.
    decode_kernel: str = "per_head"
    # The resolved backend for this step, and the longest row in it: a split-K or flash
    # kernel needs the bound host-side, since reading `seq_lens` would synchronize.
    decode_backend: object = None
    decode_max_len: int = 0
    # INT8 pools only: per-layer `[kv_heads, head_dim]` static key smoothing, or None.
    key_smoothing: list | None = None
    # INT8 pools only: use the offline-searched decode launch table (see
    # `_int8_decode_launch`) instead of the per-head T4 regime.
    int8_decode_tuned: bool = False


_BATCH_CTX: Optional[_BatchContext] = None
@dataclass
class _PrefillContext:
    """Paged-pool metadata for one mixed-length prefill chunk batch."""
    key_pool: list
    value_pool: list
    block_tables: torch.Tensor
    start_positions: torch.Tensor
    chunk_lens: torch.Tensor
    key_scale_pool: list | None = None
    value_scale_pool: list | None = None
    # Name of the resolved prefill backend, kept for graph keys and reporting.
    attention: str = "sdpa"
    # The backend itself; set by `_set_prefill_context`.
    backend: object = None
    # Longest start + chunk in this batch, known host-side at planning time. The SDPA
    # path gathers this many logical tokens per row; reading it from the device tensors
    # would be a synchronization per layer.
    total_len: int = 0
    prefill_block_m: int | None = None
    prefill_block_n: int | None = None
    key_smoothing: list | None = None
    # INT8 pools only: "per_token" (one program per query token) or "tiled" (tensor cores).
    int8_prefill_attention: str = "per_token"
    # SDPA path: the causal mask and the page-index vector depend only on the batch's
    # metadata, not the layer. Built on the first layer and reused by the other 27; each
    # rebuild was ~15 small launches, which in a launch-bound forward cost more than the
    # attention kernel saved.
    sdpa_cache: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sdpa_cache is None:
            self.sdpa_cache = {}

    @property
    def tiled_prefill(self) -> bool:
        return self.attention == "tiled"


_PREFILL_CTX: Optional[_PrefillContext] = None


@dataclass
class _FusedContext:
    """Token layout of one fused step forward: `[1, decode_rows + prefill_rows * width]`.

    The decode rows come first, one token each, then the prefill chunk batch flattened
    row-major at the staged chunk width. The per-row metadata lives in `_BATCH_CTX` and
    `_PREFILL_CTX` exactly as for the separate forwards; this only says where to cut.
    """
    decode_rows: int
    prefill_rows: int
    width: int
    # Packed layout (opt-in): the chunk tokens follow the decode tokens back to back,
    # `packed_tokens` long (the prefill budget), instead of `prefill_rows * width`. Only
    # attention needs the padded `[rows, width]` shape; it is built with `to_padded`
    # (padded slot -> packed index) and undone with `to_packed` (packed index -> slot).
    packed_tokens: int = 0
    to_padded: torch.Tensor | None = None
    to_packed: torch.Tensor | None = None


_FUSED_CTX: Optional[_FusedContext] = None
_ATTN_CALLS = 0


def _set_batch_ctx(ctx: _BatchContext) -> None:
    global _BATCH_CTX
    _BATCH_CTX = ctx


def _clear_batch_ctx() -> None:
    global _BATCH_CTX
    _BATCH_CTX = None


def _set_prefill_ctx(ctx: _PrefillContext) -> None:
    global _PREFILL_CTX
    _PREFILL_CTX = ctx


def _clear_prefill_ctx() -> None:
    global _PREFILL_CTX
    _PREFILL_CTX = None


def _set_fused_ctx(ctx: _FusedContext) -> None:
    global _FUSED_CTX
    _FUSED_CTX = ctx


def _clear_fused_ctx() -> None:
    global _FUSED_CTX
    _FUSED_CTX = None


def _decode_attention(ctx: _BatchContext, layer_idx: int, query, key, value, scaling):
    """Write one new K/V per row, then attend `[N, heads, 1, D]` over each row's pages."""
    key_pool = ctx.key_pool[layer_idx]      # [num_blocks, block_size, kv_heads, D]
    value_pool = ctx.value_pool[layer_idx]
    if ctx.key_scale_pool is None:
        write_decode_kv(key, value, key_pool, value_pool, ctx.block_tables, ctx.seq_lens)
        return ctx.decode_backend.run(
            query, key_pool, value_pool, ctx.block_tables, ctx.seq_lens,
            scale=scaling, block_n=ctx.decode_block_n, num_warps=ctx.decode_num_warps,
            length_offset=1, max_sequence_length=ctx.decode_max_len,
            num_splits=ctx.decode_num_splits,
        )
    from engine.kernels.int8_paged_kv import paged_decode_batched_int8, write_decode_int8_kv
    smoothing = ctx.key_smoothing[layer_idx] if ctx.key_smoothing is not None else None
    write_decode_int8_kv(
        key, value, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
        ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.seq_lens,
        key_smoothing=smoothing,
    )
    if ctx.int8_decode_tuned:
        launch = _int8_decode_launch(query.shape[0], ctx.decode_block_n)
        if launch is not None:
            from engine.kernels.int8_decode_tuned import paged_decode_int8_per_head
            block_n, num_warps, num_stages = launch
            return paged_decode_int8_per_head(
                query, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
                ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.seq_lens,
                scale=scaling, block_n=block_n, num_warps=num_warps, num_stages=num_stages,
                length_offset=1, key_smoothing=smoothing,
            )
    return paged_decode_batched_int8(
        query, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
        ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.seq_lens,
        scale=scaling, block_n=ctx.decode_block_n, num_warps=ctx.decode_num_warps,
        length_offset=1, key_smoothing=smoothing,
    )


def _int8_decode_launch(rows: int, regime_block_n: int) -> tuple[int, int, int] | None:
    """RTX 4060 INT8 decode launch from `int8_decode_search.json`, or None for the default.

    Keyed only on the launched row count (the graph bucket) and the existing tile regime,
    both part of every decode graph key, so a captured graph never changes launch on
    replay. Batch 1 is latency-bound (16 programs on 24 SMs): 256-token tiles halve the
    dependent iterations, 1.07-1.24x. Batch 8 runs below the bandwidth roofline with
    128x4; 64x2 is 1.11-1.15x. Batch 4 and 16 already sit at ~240 GB/s and keep the default.
    """
    if regime_block_n != 128:
        return None
    if rows == 1:
        return 256, 4, 3
    if _INT8_DECODE_B8_RULE and 7 <= rows <= 12:
        return 64, 2, 2
    return None


# Isolated search: batch 8 64x2 1.11-1.15x; end to end (chat, max_active 8) the decode
# step regressed +2.7%. Kept switchable only to reproduce that A/B; off by default.
_INT8_DECODE_B8_RULE = False


def batched_decode_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs,
):
    """K4 attention for a batched decode step.

    query/key/value: [N, heads, 1, D]  (N sequences, 1 new token each, already RoPE'd)
    Writes each sequence's new K,V into the pool, then runs K4 over each sequence's blocks.
    """
    global _ATTN_CALLS
    _ATTN_CALLS += 1

    ctx = _BATCH_CTX
    assert ctx is not None, "batched decode context not set"
    N, num_q_heads, one, D = query.shape
    assert one == 1, "batched decode: 1 new token per sequence"
    out = _decode_attention(ctx, module.layer_idx, query, key, value, scaling)
    # HF expects [N, 1, heads, D] (transposed form)
    out = out.transpose(1, 2).contiguous()   # [N, 1, num_q_heads, D]
    return out, None


def _prefill_attention(ctx: _PrefillContext, layer_idx: int, query, key, value, scaling):
    """Write a padded K/V chunk batch, then attend `[B, heads, Q, D]` to each paged prefix."""
    key_pool = ctx.key_pool[layer_idx]
    value_pool = ctx.value_pool[layer_idx]
    if ctx.key_scale_pool is None:
        from engine.kernels.kv_write import write_prefill_kv_batched
        write_prefill_kv_batched(
            key, value, key_pool, value_pool, ctx.block_tables,
            ctx.chunk_lens, ctx.start_positions,
        )
        out = ctx.backend.run(
            query, key_pool, value_pool, ctx.block_tables,
            ctx.start_positions, ctx.chunk_lens, scale=scaling,
            total_len=ctx.total_len, cache=ctx.sdpa_cache,
            block_m=ctx.prefill_block_m, block_n=ctx.prefill_block_n,
        )
    else:
        from engine.kernels.int8_paged_kv import paged_prefill_int8, write_prefill_int8_kv_batched
        smoothing = ctx.key_smoothing[layer_idx] if ctx.key_smoothing is not None else None
        write_prefill_int8_kv_batched(
            key, value, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
            ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.chunk_lens,
            ctx.start_positions, key_smoothing=smoothing,
        )
        if ctx.int8_prefill_attention == "tiled":
            from engine.kernels.int8_paged_kv import tiled_paged_prefill_int8 as int8_prefill
        else:
            int8_prefill = paged_prefill_int8
        out = int8_prefill(
            query, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
            ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.start_positions,
            ctx.chunk_lens, scale=scaling, key_smoothing=smoothing,
        )
    return out


def chunked_prefill_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs,
):
    """Write a K/V chunk, then attend it to the paged prefix causally."""
    ctx = _PREFILL_CTX
    assert ctx is not None, "chunked prefill context not set"
    out = _prefill_attention(ctx, module.layer_idx, query, key, value, scaling)
    return out.transpose(1, 2).contiguous(), None


def fused_step_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs,
):
    """Attention for one forward that carries a decode batch and a prefill chunk batch.

    The model sees a single packed row: `query` is `[1, heads, T, D]` with the decode
    tokens first and the chunk batch after them, so every per-token module (norms,
    projections, MLP) runs once over both. Only attention cares which token belongs to
    which request; this cuts the packed row into the two shapes the paged kernels take
    and puts the results back in packed order. The cuts are contiguous copies of a few
    hundred kilobytes per layer, which is well under the second forward they replace.
    """
    global _ATTN_CALLS
    _ATTN_CALLS += 1

    fused = _FUSED_CTX
    assert fused is not None, "fused step context not set"
    layer_idx = module.layer_idx
    one, heads, total, head_dim = query.shape
    assert one == 1, "fused step packs every token into one row"
    decode_rows, prefill_rows, width = fused.decode_rows, fused.prefill_rows, fused.width
    packed = fused.packed_tokens > 0
    prefill_tokens = fused.packed_tokens if packed else prefill_rows * width
    assert total == decode_rows + prefill_tokens, "fused row layout mismatch"
    parts = []
    if decode_rows:
        ctx = _BATCH_CTX
        assert ctx is not None, "batched decode context not set"
        # [1, heads, N, D] -> [N, heads, 1, D]
        cut = slice(0, decode_rows)
        out = _decode_attention(
            ctx, layer_idx,
            query[:, :, cut].transpose(0, 2).contiguous(),
            key[:, :, cut].transpose(0, 2).contiguous(),
            value[:, :, cut].transpose(0, 2).contiguous(),
            scaling,
        )
        parts.append(out.transpose(1, 2).reshape(1, decode_rows, heads, head_dim))
    if prefill_rows:
        ctx = _PREFILL_CTX
        assert ctx is not None, "chunked prefill context not set"

        def rows(tensor):
            # [1, h, B*W, D] -> [B, h, W, D]; packed: [1, h, P, D] -> gather -> [B, h, W, D]
            h = tensor.shape[1]
            chunk = tensor[0, :, decode_rows:]
            if packed:
                chunk = chunk.index_select(1, fused.to_padded)
            return chunk.reshape(h, prefill_rows, width, head_dim).transpose(0, 1).contiguous()

        out = _prefill_attention(ctx, layer_idx, rows(query), rows(key), rows(value), scaling)
        out = out.transpose(1, 2).reshape(prefill_rows * width, heads, head_dim)
        if packed:
            out = out.index_select(0, fused.to_packed)
        parts.append(out.reshape(1, prefill_tokens, heads, head_dim))
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    return out.contiguous(), None


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def _awq_prefill_on_triton(model, chunk_rows: int) -> bool:
    """True when every AWQ projection runs a full prefill chunk on the Triton int4pack GEMM."""
    from engine.quantization.awq_tinygemm import AWQTinyGemmLinear

    thresholds = [module.triton_prefill_rows for module in model.modules()
                  if isinstance(module, AWQTinyGemmLinear)]
    return bool(thresholds) and all(t is not None and t <= chunk_rows for t in thresholds)


class ContinuousBatchingEngine:
    """Full batched decode over paged KV, scheduler-driven."""

    ATTN_NAME = "batched_paged_decode"
    PREFILL_ATTN_NAME = "chunked_paged_prefill"
    FUSED_ATTN_NAME = "fused_paged_decode_prefill"
    # Gate 1B removed the preemption-count limit. A count is the wrong criterion: a
    # healthy request waiting behind several long generations yields once per iteration
    # through no fault of its own, so any fixed bound fails valid work under load.
    # Termination is now structural - see FCFSScheduler's policy docstring.

    def __init__(self, model, tokenizer, device, *,
                 num_blocks: int = 4096, block_size: int = 16, max_active: int = 16,
                 prefill_chunk_size: int = 128,
                 prefill_attention: str | None = None,
                 decode_attention: str = "per_head",
                 tiled_prefill: bool = False,
                 prefill_block_m: int | None = None,
                 prefill_block_n: int | None = None,
                 decode_num_splits: int = 0,
                 max_prefill_tokens_per_iteration: int = 128,
                 idle_prefill_tokens: int | None = 4096,
                 max_waiting_requests: int | None = None,
                 prefix_cache_blocks: int = 256,
                 kv_cache_dtype: str = "fp16",
                 kv_key_smoothing: torch.Tensor | None = None,
                 int8_prefill_attention: str = "per_token",
                 int8_decode_tuned: bool = False,
                 cuda_graph_batch_size: int | None = None,
                 cuda_graph_batch_sizes: tuple[int, ...] | None = None,
                 prefill_cuda_graphs: bool = True,
                 fused_step: bool = True,
                 packed_fused_prefill: bool = False,
                 packed_prefill_width_buckets: bool = False,
                 fuse_mlp_gate_up: bool = False,
                 fuse_qkv: bool = False,
                 triton_rmsnorm: bool = True,
                 triton_rope: bool = True,
                 triton_swiglu: bool = True,
                 sampling_seed: int | None = None):
        if min(num_blocks, block_size, max_active, prefill_chunk_size,
               max_prefill_tokens_per_iteration) <= 0:
            raise ValueError("engine sizes and prefill budgets must be positive")
        if prefix_cache_blocks < 0:
            raise ValueError("prefix_cache_blocks must be non-negative")
        if kv_cache_dtype not in {"fp16", "int8"}:
            raise ValueError("kv_cache_dtype must be 'fp16' or 'int8'")
        if kv_key_smoothing is not None and kv_cache_dtype != "int8":
            raise ValueError("kv_key_smoothing applies only to kv_cache_dtype='int8'")
        if int8_prefill_attention not in {"per_token", "tiled"}:
            raise ValueError("int8_prefill_attention must be 'per_token' or 'tiled'")
        # Chunked-prefill attention over an INT8 pool. "tiled" is the tensor-core kernel
        # selected by the offline RTX 4060 search; opt-in until its end-to-end A/B.
        self.int8_prefill_attention = int8_prefill_attention
        self.int8_decode_tuned = int8_decode_tuned
        if decode_num_splits < 0:
            raise ValueError("decode_num_splits must be non-negative")
        if cuda_graph_batch_size is not None and cuda_graph_batch_sizes is not None:
            raise ValueError("use cuda_graph_batch_size or cuda_graph_batch_sizes, not both")
        if cuda_graph_batch_sizes is None and cuda_graph_batch_size is not None:
            cuda_graph_batch_sizes = (cuda_graph_batch_size,)
        if cuda_graph_batch_sizes is not None:
            cuda_graph_batch_sizes = tuple(sorted(set(cuda_graph_batch_sizes)))
            if not cuda_graph_batch_sizes or any(not 0 < size <= max_active for size in cuda_graph_batch_sizes):
                raise ValueError("graph bucket sizes must be within [1, max_active]")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.max_active = max_active
        self.prefill_chunk_size = prefill_chunk_size
        cfg = model.config
        # Attention backends are resolved against this device and this geometry rather
        # than hard-coded: "auto" (the default) takes the measured default for the
        # architecture, or the highest-priority runnable backend on one nobody has
        # measured. A named backend that cannot run here raises, with the reason - a
        # silent fallback is how a week of T4 results measured the wrong path.
        # `tiled_prefill=True` is the older boolean spelling of `prefill_attention="tiled"`.
        from engine.kernels.device import current_device

        config_geometry = Geometry(
            num_q_heads=int(cfg.num_attention_heads),
            num_kv_heads=int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
            head_dim=int(getattr(cfg, "head_dim", 0)
                         or cfg.hidden_size // cfg.num_attention_heads),
            block_size=block_size,
            dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
            kv_dtype="int8" if kv_cache_dtype == "int8" else "float16",
        )
        self.device_profile = current_device()
        self.geometry = config_geometry
        device_defaults = defaults_for(self.device_profile, config_geometry)
        self.backend_reasons = dict(device_defaults.reasons)
        if prefill_attention is None:
            prefill_attention = "tiled" if tiled_prefill else device_defaults.prefill_attention
        if decode_attention in (None, "auto"):
            decode_attention = device_defaults.decode_attention
        self.prefill_backend = resolve(
            "prefill", prefill_attention, self.device_profile, config_geometry,
        )
        self.decode_backend = resolve(
            "decode", decode_attention, self.device_profile, config_geometry,
        )
        self.prefill_attention = self.prefill_backend.name
        self.decode_attention = self.decode_backend.name
        self.tiled_prefill = self.prefill_attention == "tiled"
        self.prefill_block_m = prefill_block_m
        self.prefill_block_n = prefill_block_n
        self.decode_num_splits = decode_num_splits
        self.max_prefill_tokens_per_iteration = max_prefill_tokens_per_iteration
        # With no request decoding there is no ITL to protect, so fresh prompts are
        # prefilled whole (up to this many tokens per step) through the single-forward
        # fast path instead of `prefill_chunk_size` slices: a 2048-token prompt took
        # sixteen 128-token steps and a longer TTFT than stock Transformers. None keeps
        # chunking always. The chunk budget still applies whenever anything decodes.
        if idle_prefill_tokens is not None and idle_prefill_tokens <= 0:
            raise ValueError("idle_prefill_tokens must be positive or None")
        self.idle_prefill_tokens = idle_prefill_tokens
        self.max_waiting_requests = max_waiting_requests
        self.prefix_cache_blocks = prefix_cache_blocks
        self.kv_cache_dtype = kv_cache_dtype
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or ()
        self.decode_cuda_graphs = self.decode_backend.graph_safe
        self.decode_graph_eager_reason: str | None = None
        if not self.decode_cuda_graphs:
            # The FA2 kvcache entry point synchronizes while preparing its split-KV
            # workspace, including with an explicit split count, and invalidates stream
            # capture on the tested 2.8.4 build. Keep other phase graphs enabled.
            self.decode_graph_eager_reason = (
                "flash_attn kvcache decode is eager-only on this build: its split-KV "
                "workspace setup is not CUDA-graph safe"
            )
        # Chunked prefill forwards are captured per (row bucket, kind, context bucket)
        # when graph buckets are configured. Off keeps decode graphs and runs prefill
        # eagerly on the same staged buffers, which is the A/B for the capture itself.
        self.prefill_cuda_graphs = prefill_cuda_graphs
        self.prefill_graph_eager_reason: str | None = None
        if (getattr(model.config, "_weight_only_int4_backend", None) == "tinygemm"
                and not (packed_fused_prefill and packed_prefill_width_buckets
                         and _awq_prefill_on_triton(model, prefill_chunk_size))):
            # On RTX 4060, native INT4 prefill capture doubled short-prompt TTFT
            # (108-146 ms eager versus ~222 ms captured) without changing ITL. With the
            # Triton int4pack prefill GEMM, capture still raised chat ITL p99 42% with
            # padded rows and was inconclusive with full-width packed rows. Only packed
            # rows with width buckets won (ITL p999 -34%, every TTFT pair lower;
            # ab_awq_graphs_packed_buckets_chat_qwen3_4b.json), so only that opt-in
            # combination captures AWQ prefill.
            self.prefill_cuda_graphs = False
            self.prefill_graph_eager_reason = (
                "native INT4 weight-only models use measured eager prefill; captured "
                "tinygemm regressed RTX 4060 TTFT"
            )
        elif not self.prefill_backend.graph_safe:
            # FA2 kvcache has no query-length vector. Correct variable-size chunks are
            # grouped by their active length in `flash_paged_prefill`, which requires a
            # host-side grouping step; its internal workspace allocation is also not
            # capture-safe on the tested build. Do not capture an all-padding warmup
            # graph and replay it for real rows: Flash prefill is deliberately eager.
            self.prefill_cuda_graphs = False
            self.prefill_graph_eager_reason = (
                f"{self.prefill_attention} prefill backend is eager-only: variable "
                "chunk groups and/or backend workspace are not CUDA-graph safe"
            )
        # A step that carries both decode rows and prefill chunks runs them as one packed
        # forward (decode tokens first, chunk batch after) instead of two. The weights
        # are read once for both, and the prefill's launch cost rides on the decode
        # forward's. Off runs the two forwards back to back, which is the A/B.
        self.fused_step = fused_step
        # Packed fused prefill: the fused forward carries `budget` chunk tokens however
        # the budget is split across rows, instead of every row padded to the full chunk.
        # A graph replay of two 40/88-token chunks otherwise ran 256 tokens through every
        # projection and MLP (RTX 4060, Qwen3-1.7B: 63-68 ms versus 41 ms for one row).
        self.packed_fused_prefill = packed_fused_prefill
        # Packed fused graphs are otherwise shaped for the whole budget, so a step that plans
        # 40 chunk tokens still runs `budget` through every projection. Width buckets
        # (powers of two from 32 up to the budget) replay the smallest graph that fits.
        self.packed_prefill_width_buckets = packed_prefill_width_buckets
        self.fuse_mlp_gate_up = fuse_mlp_gate_up
        self.fuse_qkv = fuse_qkv
        self.triton_rmsnorm = triton_rmsnorm
        self.triton_rope = triton_rope
        self.triton_swiglu = triton_swiglu
        self._decode_graphs = {}

        # Refuse a checkpoint whose geometry the paged kernels cannot serve, at load,
        # with the reason - rather than a wrong answer or a launch failure later.
        from engine.model.adapters import ensure_supported

        ensure_supported(cfg)
        self.num_layers = cfg.num_hidden_layers
        self.max_model_len = getattr(cfg, "max_position_embeddings", None)
        # Step accounting: a prefill-carrying iteration and a decode-only iteration cost
        # very different amounts, and mixing them makes any latency percentile a blend.
        self.prefill_steps = 0
        self.decode_only_steps = 0
        # Of the prefill-carrying steps, how many ran decode and prefill as one forward.
        self.fused_steps = 0
        self.last_step_prefill_tokens = 0
        self.last_step_decode_rows = 0
        # Which prefill implementation ran. The SDPA fast path and the resumable chunk
        # path have different cost structures, so a measured prefill cost cannot be
        # attributed to either without knowing which one produced it.
        self.prefill_sdpa_calls = 0
        self.prefill_chunked_calls = 0
        self.prefill_sdpa_tokens = 0
        self.prefill_chunked_tokens = 0
        self.last_step_prefill_path = ""
        # Opt-in step decomposition. A wall-clock timer around `step()` blends Python
        # staging, the H2D copies, the forward and the sampling sync, so it cannot say
        # which of them a change moved. With `instrument` on, `last_step_timing` holds
        # per-phase milliseconds for the step just run: `host_stage_ms` (metadata staging
        # before the copies), `decode_gpu_ms` / `prefill_gpu_ms` (CUDA events around each
        # forward), `sync_ms` (the sampling device-to-host wait). GPU phases need an event
        # sync, which decode already pays at sampling; a prefill step that completes no
        # request gains one sync it would not otherwise have, so leave this off for
        # production serving and on for benchmarks that want the split.
        self.instrument = False
        self.last_step_timing: dict[str, float] = {}
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.num_q_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

        # Qwen uses RMSNorm for hidden states and for per-head Q/K normalization.
        # Install one FP32-accumulating Triton kernel for both shapes before warmup.
        # Each fusion is a toggle so an A/B can attribute it: a `False` flag actively
        # restores the stock implementation, because benchmarks build several engines
        # on one shared model object and a previous engine may have patched it.
        from engine.kernels.rmsnorm import install_triton_rmsnorm, uninstall_triton_rmsnorm
        from engine.kernels.rope import install_triton_rope, uninstall_triton_rope
        from engine.kernels.swiglu import install_triton_swiglu, uninstall_triton_swiglu
        from engine.kernels.swiglu import install_fused_qkv, uninstall_fused_qkv
        if triton_rmsnorm:
            self.triton_rmsnorm_modules = install_triton_rmsnorm(model)
        else:
            uninstall_triton_rmsnorm(model)
            self.triton_rmsnorm_modules = 0
        # RoPE is patched in this model's own modeling module, so a family the engine has
        # never seen is covered without an entry anywhere.
        if triton_rope:
            install_triton_rope(model)
        else:
            uninstall_triton_rope(model)
        if fuse_mlp_gate_up and not triton_swiglu:
            raise ValueError("fuse_mlp_gate_up requires triton_swiglu=True")
        # Re-install when the fusion mode changes: the installer is a no-op on an
        # already patched module, so a prior engine's choice would otherwise persist.
        uninstall_triton_swiglu(model)
        if triton_swiglu:
            self.triton_swiglu_modules = install_triton_swiglu(
                model, fuse_gate_up=fuse_mlp_gate_up,
            )
        else:
            self.triton_swiglu_modules = 0
        # Q/K/V projection fusion changes only the three input GEMMs in attention;
        # cache writes, RoPE and the selected paged-attention backend stay untouched.
        # Reinstall explicitly because benchmark engines may share the same model.
        uninstall_fused_qkv(model)
        self.fused_qkv_modules = install_fused_qkv(model) if fuse_qkv else 0

        # One sampler for the engine. Requests that ask for nothing sample greedily,
        # which is the argmax path every recorded benchmark measured; `sampling_seed`
        # seeds the shared generator for requests that sample without their own seed.
        self.sampler = BatchedSampler(device, generator_seed=sampling_seed)
        self.vocab_size = int(getattr(cfg, "vocab_size", 0))

        self.eos_ids = set()
        ce = model.generation_config.eos_token_id
        if isinstance(ce, int):
            self.eos_ids.add(ce)
        elif isinstance(ce, (list, tuple)):
            self.eos_ids.update(ce)

        dtype = torch.int8 if kv_cache_dtype == "int8" else next(model.parameters()).dtype
        self.key_pool = [
            torch.zeros((num_blocks, block_size, self.num_kv_heads, self.head_dim),
                        device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self.value_pool = [torch.zeros_like(k) for k in self.key_pool]
        if kv_cache_dtype == "int8":
            scale_shape = (num_blocks, block_size, self.num_kv_heads)
            self.key_scale_pool = [torch.zeros(scale_shape, device=device, dtype=torch.float16)
                                   for _ in range(self.num_layers)]
            self.value_scale_pool = [torch.zeros_like(scale) for scale in self.key_scale_pool]
            self._int8_preemption_spills: dict[str, tuple[int, tuple[torch.Tensor, ...]]] = {}
            # Static key smoothing (see engine.quantization.kv_smoothing). Opt-in: the
            # default INT8 mode keeps plain per-token key scales.
            self.key_smoothing = None
            if kv_key_smoothing is not None:
                expected = (self.num_layers, self.num_kv_heads, self.head_dim)
                if tuple(kv_key_smoothing.shape) != expected:
                    raise ValueError(f"kv_key_smoothing must have shape {expected}")
                self.key_smoothing = [
                    layer.to(device=device, dtype=torch.float32).contiguous()
                    for layer in kv_key_smoothing
                ]
        else:
            self.key_scale_pool = None
            self.value_scale_pool = None
            self._int8_preemption_spills = {}
            self.key_smoothing = None
        self._int8_spill_events = 0
        self._int8_spill_bytes_total = 0
        self._int8_spill_write_ms_total = 0.0
        self._int8_spill_restore_ms_total = 0.0

        # Decode metadata has a fixed upper bound. Keep both pinned-host staging and
        # GPU tensors alive for the engine lifetime so each token step performs a few
        # batched copies rather than allocating tensors and launching one scalar copy
        # per block-table entry.
        self._host_input_ids = torch.empty((max_active, 1), dtype=torch.long, pin_memory=True)
        self._host_position_ids = torch.empty((max_active, 1), dtype=torch.long, pin_memory=True)
        self._host_seq_lens = torch.empty((max_active,), dtype=torch.int32, pin_memory=True)
        self._host_block_tables = torch.empty(
            (max_active, num_blocks), dtype=torch.int32, pin_memory=True
        )
        self._device_input_ids = torch.empty((max_active, 1), dtype=torch.long, device=device)
        self._device_position_ids = torch.empty((max_active, 1), dtype=torch.long, device=device)
        self._device_seq_lens = torch.empty((max_active,), dtype=torch.int32, device=device)
        self._device_block_tables = torch.empty(
            (max_active, num_blocks), dtype=torch.int32, device=device
        )
        # Chunked prefill has the same shape discipline as decode: every chunk batch is
        # staged into persistent buffers of fixed address, [max_active, chunk] wide, so a
        # captured forward can be replayed against them. Rows past the batch and columns
        # past a chunk are inert (chunk length 0 / pad token) and touch no KV.
        chunk = prefill_chunk_size
        self._prefill_host_input_ids = torch.empty((max_active, chunk), dtype=torch.long, pin_memory=True)
        self._prefill_host_position_ids = torch.empty((max_active, chunk), dtype=torch.long, pin_memory=True)
        self._prefill_host_starts = torch.empty((max_active,), dtype=torch.int32, pin_memory=True)
        self._prefill_host_chunk_lens = torch.empty((max_active,), dtype=torch.int32, pin_memory=True)
        self._prefill_host_block_tables = torch.empty(
            (max_active, num_blocks), dtype=torch.int32, pin_memory=True
        )
        self._prefill_device_input_ids = torch.empty((max_active, chunk), dtype=torch.long, device=device)
        self._prefill_device_position_ids = torch.empty((max_active, chunk), dtype=torch.long, device=device)
        self._prefill_device_starts = torch.empty((max_active,), dtype=torch.int32, device=device)
        self._prefill_device_chunk_lens = torch.empty((max_active,), dtype=torch.int32, device=device)
        self._prefill_device_block_tables = torch.empty(
            (max_active, num_blocks), dtype=torch.int32, device=device
        )
        # Packed fused-prefill staging (see `_prepare_packed_metadata`).
        packed = max_prefill_tokens_per_iteration
        self._packed_tokens = packed
        widths = [packed]
        if packed_prefill_width_buckets:
            width_bucket = 32
            while width_bucket < packed:
                widths.append(width_bucket)
                width_bucket *= 2
        self._packed_widths = tuple(sorted(set(widths)))
        self._active_packed_width = packed
        self._packed_host_ids = torch.empty((packed,), dtype=torch.long, pin_memory=True)
        self._packed_host_positions = torch.empty((packed,), dtype=torch.long, pin_memory=True)
        self._packed_host_to_packed = torch.empty((packed,), dtype=torch.long, pin_memory=True)
        self._packed_host_to_padded = torch.empty((max_active * chunk,), dtype=torch.long, pin_memory=True)
        self._packed_host_last = torch.empty((max_active,), dtype=torch.long, pin_memory=True)
        self._packed_device_ids = torch.empty((packed,), dtype=torch.long, device=device)
        self._packed_device_positions = torch.empty((packed,), dtype=torch.long, device=device)
        self._packed_device_to_packed = torch.empty((packed,), dtype=torch.long, device=device)
        self._packed_device_to_padded = torch.empty((max_active * chunk,), dtype=torch.long, device=device)
        self._packed_device_last = torch.empty((max_active,), dtype=torch.long, device=device)
        # Captured prefill and fused graphs end in the vocabulary projection, and their
        # output has to live at a fixed address for replay. A private `[rows, vocab]`
        # tensor per graph would be ~5 MB each and there are dozens of shapes, so every
        # graph copies into this one buffer instead. Decode graphs keep the model's own
        # logits tensor, of which there are only a handful.
        # A fused step needs one row per decode row plus one per chunk row, and both
        # are bounded by `max_active`.
        self._logits_buffer = torch.empty(
            (2 * max_active, self.vocab_size), dtype=next(model.parameters()).dtype,
            device=device,
        ) if self.vocab_size else None
        self._live_row_index: dict[tuple[int, int, int], torch.Tensor] = {}
        self._prefill_graphs = {}
        self._fused_graphs = {}
        self._prefill_graph_pool = None
        # Graph captures taken while serving, i.e. shapes warmup did not cover. Each one
        # is two eager forwards plus device syncs inside a live step (~100-200 ms on the
        # T4), so a non-zero count after warmup is a tail-latency finding, not a detail.
        self.lazy_graph_captures = 0
        # Attention kinds whose forward could not be captured on this build; they run
        # eagerly with the same staged buffers. Recorded once, with the reason.
        self._prefill_graph_unsupported: dict[str, str] = {}
        self.block_manager = KVBlockManager(
            num_blocks=num_blocks, block_size_tokens=block_size
        )
        self._pool_token_capacity = num_blocks * block_size
        self._reserve_graph_dummy_blocks()
        self.prefix_cache = PrefixCache(self.block_manager, prefix_cache_blocks)
        self.scheduler = FCFSScheduler(
            self.block_manager, max_waiting_requests=max_waiting_requests,
            prefix_cache=self.prefix_cache,
            # Dummy rows are never returned to the pool, so admission must not count them.
            reserved_blocks=len(self._graph_dummy_blocks),
        )

        # Register the batched attention fn once.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS[self.ATTN_NAME] = batched_decode_attention_forward
        ALL_ATTENTION_FUNCTIONS[self.PREFILL_ATTN_NAME] = chunked_prefill_attention_forward
        ALL_ATTENTION_FUNCTIONS[self.FUSED_ATTN_NAME] = fused_step_attention_forward

    def _reserve_graph_dummy_blocks(self) -> None:
        """Reserve permanent, non-customer pages for padded CUDA-Graph rows."""
        self._graph_dummy_blocks: list[int] = []
        if not self.cuda_graph_batch_sizes:
            return
        # Padded rows need distinct dummy pages; capture on zero live rows (fused graphs)
        # needs at least one even when the largest bucket is 1, or every fused capture
        # fails and a max_active=1 engine runs its fused steps eagerly.
        required = max(1, max(self.cuda_graph_batch_sizes) - 1)
        allocation = self.block_manager.reserve("__cuda_graph_dummy_rows__", required * self.block_size)
        if allocation is None:
            raise ValueError("insufficient KV blocks to reserve CUDA-Graph dummy rows")
        self._graph_dummy_blocks = allocation.physical_block_ids

    def _prepare_decode_metadata(
        self, active: list[GenerationRequest], *, graph_bucket_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage one compact decode batch into persistent GPU metadata buffers."""
        count = len(active)
        row_count = graph_bucket_size or count
        if count > self.max_active:
            raise ValueError("active batch exceeds max_active")
        if row_count < count or row_count > self.max_active:
            raise ValueError("invalid graph bucket size")
        # Gather the batch as Python lists first and stage each buffer with one slice
        # copy. A tensor `__setitem__` per element costs a few microseconds of dispatch,
        # and the block-table loop alone ran to several milliseconds per step at batch 16
        # with 1k-token contexts - all of it serialized ahead of the graph replay.
        token_ids: list[int] = []
        lengths: list[int] = []
        for request in active:
            allocation = request.allocation
            if allocation is None or request.next_token_id is None:
                raise RuntimeError("decode request is missing token or KV allocation")
            if allocation.sequence_length >= allocation.capacity_tokens:
                # The write kernel would otherwise index past this row's block table.
                raise RuntimeError(
                    f"request {request.request_id!r} has no KV slot for its next token"
                )
            token_ids.append(request.next_token_id)
            lengths.append(allocation.sequence_length)

        if row_count > count:
            if not self._graph_dummy_blocks or (
                    count and len(self._graph_dummy_blocks) < row_count - count):
                raise RuntimeError("graph dummy blocks were not reserved for this bucket")
            pad_token = self.tokenizer.pad_token_id
            if pad_token is None:
                pad_token = next(iter(self.eos_ids), 0)
            token_ids.extend([pad_token] * (row_count - count))
            lengths.extend([0] * (row_count - count))

        self._host_input_ids[:row_count, 0] = torch.tensor(token_ids, dtype=torch.long)
        self._host_position_ids[:row_count, 0] = torch.tensor(lengths, dtype=torch.long)
        self._host_seq_lens[:row_count] = torch.tensor(lengths, dtype=torch.int32)
        for row, request in enumerate(active):
            table = request.block_table
            self._host_block_tables[row, :len(table)] = torch.tensor(table, dtype=torch.int32)
        # A live batch has at least one real row, so its inert rows get distinct dummy
        # pages. A capture on no rows at all (the fused step graphs) wraps around: two
        # inert rows then store the same pad token into the same slot, which is benign.
        dummies = self._graph_dummy_blocks
        for row in range(count, row_count):
            self._host_block_tables[row, 0] = dummies[(row - count) % len(dummies)]

        input_ids = self._device_input_ids[:row_count]
        position_ids = self._device_position_ids[:row_count]
        seq_lens = self._device_seq_lens[:row_count]
        # Keep the complete row width so this view is contiguous. The Triton kernel
        # indexes only blocks covered by seq_lens; unused columns are never read.
        block_tables = self._device_block_tables[:row_count]
        input_ids.copy_(self._host_input_ids[:row_count], non_blocking=True)
        position_ids.copy_(self._host_position_ids[:row_count], non_blocking=True)
        seq_lens.copy_(self._host_seq_lens[:row_count], non_blocking=True)
        block_tables.copy_(self._host_block_tables[:row_count], non_blocking=True)
        return input_ids, position_ids, block_tables, seq_lens

    def reset(self) -> None:
        """Reinitialize the allocator (fresh free-block list) for a clean run.

        Blocks are already released as sequences finish, but this guarantees a clean
        slate when reusing the same engine for multiple benchmark runs. The pool tensors
        are reused (not reallocated) — only the allocator's bookkeeping resets.
        """
        self.block_manager = KVBlockManager(
            num_blocks=self.key_pool[0].shape[0], block_size_tokens=self.block_size,
        )
        self._reserve_graph_dummy_blocks()
        self.prefix_cache = PrefixCache(self.block_manager, self.prefix_cache_blocks)
        self.scheduler = FCFSScheduler(
            self.block_manager, max_waiting_requests=self.max_waiting_requests,
            prefix_cache=self.prefix_cache,
            reserved_blocks=len(self._graph_dummy_blocks),
        )
        self._int8_preemption_spills.clear()
        self._int8_spill_events = 0
        self._int8_spill_bytes_total = 0
        self._int8_spill_write_ms_total = 0.0
        self._int8_spill_restore_ms_total = 0.0

    def _ensure_writable_tail(self, request: GenerationRequest) -> bool:
        """Copy a shared partial tail before decode writes into its unused slots."""
        allocation = request.allocation
        if allocation is None or not allocation.sequence_length % self.block_size:
            return True
        tail = allocation.physical_block_ids[-1]
        if self.block_manager.allocator.refcount(tail) <= 1:
            return True
        if self.block_manager.allocator.free_block_count == 0:
            self.prefix_cache.evict_until_free(1)
        if self.block_manager.allocator.refcount(tail) <= 1:
            return True
        copied = self.block_manager.copy_on_write_tail(request.request_id)
        if copied is None:
            return False
        old_block, new_block = copied
        # One block per layer for K and V (plus scales for INT8). Issued as a single
        # foreach copy rather than 2*num_layers separate launches: the bytes are the same,
        # the launch overhead is not, and this runs on a request's first decode step.
        pools = self.key_pool + self.value_pool
        if self.key_scale_pool is not None:
            pools = pools + self.key_scale_pool + self.value_scale_pool
        torch._foreach_copy_(
            [pool[new_block] for pool in pools], [pool[old_block] for pool in pools],
        )
        return True

    def _ensure_kv_capacity(self, request: GenerationRequest, target_length: int) -> bool:
        allocation = request.allocation
        if allocation is None:
            return False
        if target_length > allocation.sequence_length and not self._ensure_writable_tail(request):
            return False
        blocks_needed = (target_length + self.block_size - 1) // self.block_size
        extra_blocks = max(0, blocks_needed - len(allocation.physical_block_ids))
        if extra_blocks > self.block_manager.allocator.free_block_count:
            self.prefix_cache.evict_until_free(extra_blocks)
        return self.block_manager.ensure_capacity(request.request_id, target_length)

    def _acquire_capacity(self, request: GenerationRequest, target_length: int) -> bool:
        """Obtain KV capacity for ``request``, yielding newer requests if needed.

        Strict LIFO: victims are always later arrivals than the requester, so the oldest
        active request is never displaced and always completes. Returns False when the
        request cannot be served this iteration, having either yielded itself (state
        WAITING, retried after the progress epoch advances) or exhausted the options -
        in which case it is the sole active request and the caller must fail it.

        Each loop turn strictly shrinks the active set, so the loop always terminates.
        """
        while True:
            if self._ensure_kv_capacity(request, target_length):
                return True
            victim = self.scheduler.newest_active()
            if victim is None:
                return False
            if victim.request_id == request.request_id:
                # The requester is itself the correct LIFO victim. Yielding only helps if
                # an older request remains to free memory; alone, the pool genuinely
                # cannot serve it and yielding would spin forever.
                if len(self.scheduler.active) <= 1:
                    return False
                self._preempt_request(request)
                return False
            self._preempt_request(victim)

    def _preempt_request(self, request: GenerationRequest) -> None:
        """Yield a request, preserving exact INT8 pages once generation has begun.

        Recomputing a generated prefix can select a different GEMM shape from its
        original mixed batch. The resulting small hidden-state change is amplified by
        per-token INT8 quantization and can flip a tied greedy logit. Host spill is rare
        and slower than recompute, but it preserves the authoritative cache bytes.
        """
        allocation = request.allocation
        if (
            self.kv_cache_dtype == "int8" and request.output_token_ids
            and allocation is not None and request.request_id not in self._int8_preemption_spills
        ):
            spill_started = perf_counter()
            # Only the pages covering committed tokens. A decode row may already hold the
            # page for its next token when a later prefill in the same step preempts it;
            # restore allocates for the committed length, so an extra page here would
            # make the payload one page longer than its destination.
            committed = -(-allocation.sequence_length // self.block_size)
            block_ids = torch.tensor(
                allocation.physical_block_ids[:committed], dtype=torch.long, device=self.device,
            )
            # Pageable `.cpu()` is deliberate: on the measured RTX 4060 host (PCIe link
            # trained at Gen4 x1, ~1.4 GB/s) pinned per-layer async copies moved bytes
            # only ~10% faster and paid a fresh pinned allocation per spill, doubling the
            # measured write time (results/rtx4060/int8_kv_spill_pinned_ab_long_decode.json).
            payload = tuple(
                torch.stack([pool.index_select(0, block_ids) for pool in group]).cpu()
                for group in (
                    self.key_pool, self.value_pool, self.key_scale_pool, self.value_scale_pool,
                )
            )
            self._int8_preemption_spills[request.request_id] = (
                allocation.sequence_length, payload,
            )
            self._int8_spill_events += 1
            self._int8_spill_bytes_total += sum(
                tensor.numel() * tensor.element_size() for tensor in payload
            )
            self._int8_spill_write_ms_total += (perf_counter() - spill_started) * 1000
        self.scheduler.preempt(request.request_id)

    def _restore_int8_spill(self, request: GenerationRequest) -> bool:
        spill = self._int8_preemption_spills.get(request.request_id)
        if spill is None:
            return False
        sequence_length, payload = spill
        if not self._capacity_or_fail(request, sequence_length):
            if request.done:
                self._int8_preemption_spills.pop(request.request_id, None)
            return True
        allocation = request.allocation
        if allocation is None:
            return True
        restore_started = perf_counter()
        block_ids = torch.tensor(allocation.physical_block_ids, dtype=torch.long, device=self.device)
        if len(block_ids) != payload[0].shape[1]:
            raise RuntimeError(
                f"INT8 spill for {request.request_id!r} holds {payload[0].shape[1]} pages "
                f"but its restored allocation has {len(block_ids)}"
            )
        groups = (self.key_pool, self.value_pool, self.key_scale_pool, self.value_scale_pool)
        for pools, saved in zip(groups, payload):
            for layer, pool in enumerate(pools):
                pool.index_copy_(0, block_ids, saved[layer].to(self.device))
        torch.cuda.synchronize(self.device)
        self._int8_spill_restore_ms_total += (perf_counter() - restore_started) * 1000
        if not self.block_manager.append_tokens(request.request_id, sequence_length):
            raise RuntimeError("INT8 spill capacity was acquired but could not be restored")
        request.prefilled_token_count = request.prefill_token_count
        del self._int8_preemption_spills[request.request_id]
        self._complete_prefill(request, None)
        return True

    def _capacity_or_fail(self, request: GenerationRequest, target_length: int) -> bool:
        """Acquire capacity; fail the request only if preemption could not help it."""
        if self._acquire_capacity(request, target_length):
            return True
        if request.state is not RequestState.WAITING:
            self._fail_request(request, "KV_POOL_EXHAUSTED")
        return False

    def _gpu_timer(self):
        if not self.instrument or torch.device(self.device).type != "cuda":
            return None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def _gpu_elapsed(self, start, key: str) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        self.last_step_timing[key] = self.last_step_timing.get(key, 0.0) + start.elapsed_time(end)

    def _host_elapsed(self, started: float | None, key: str) -> None:
        if started is None:
            return
        self.last_step_timing[key] = (
            self.last_step_timing.get(key, 0.0) + (perf_counter() - started) * 1000
        )

    def stats_snapshot(self) -> dict[str, object]:
        """Cheap, GPU-free view of engine state for the metrics endpoint.

        Called from the worker thread only. Every value is a plain int or float so the
        result can be handed to another thread and read without touching live
        scheduler containers, which the worker mutates continuously.
        """
        blocks = self.block_manager.snapshot()
        used = int(blocks["used_blocks"])
        total = int(blocks["num_blocks"])
        reserved = len(self._graph_dummy_blocks)
        usable = max(1, total - reserved)
        cache = self.prefix_cache.snapshot()
        # The decode operating point: how many sequences a step actually advances, and how
        # much KV each carries. A roofline comparison is meaningless without both, because
        # the floor moves with context length as well as batch.
        decoding = [
            request for request in self.scheduler.active.values()
            if request.state is RequestState.DECODING and request.allocation is not None
        ]
        context_tokens = sum(r.allocation.sequence_length for r in decoding)
        spill_bytes_live = sum(
            tensor.numel() * tensor.element_size()
            for _, payload in self._int8_preemption_spills.values()
            for tensor in payload
        )
        return {
            "waiting_requests": len(self.scheduler.waiting),
            "active_requests": len(self.scheduler.active),
            "decode_batch": len(decoding),
            "decode_context_tokens": context_tokens,
            "decode_mean_context": context_tokens / len(decoding) if decoding else 0.0,
            "admitted_total": self.scheduler.admission_count,
            "rejected_total": self.scheduler.rejected_count,
            "preemptions_total": self.scheduler.preemption_count,
            "progress_epoch": self.scheduler.progress_epoch,
            "kv_blocks_total": total,
            "kv_blocks_reserved": reserved,
            "kv_blocks_used": used,
            "kv_utilization": (used - reserved) / usable,
            "prefix_cache_blocks": int(cache.get("cached_blocks", 0)),
            "prefix_cache_hits": int(cache.get("hits", 0)),
            "prefix_cache_misses": int(cache.get("lookups", 0)) - int(cache.get("hits", 0)),
            "prefill_steps": self.prefill_steps,
            "decode_only_steps": self.decode_only_steps,
            "fused_steps": self.fused_steps,
            "lazy_graph_captures": self.lazy_graph_captures,
            "prefill_sdpa_calls": self.prefill_sdpa_calls,
            "prefill_chunked_calls": self.prefill_chunked_calls,
            "prefill_sdpa_tokens": self.prefill_sdpa_tokens,
            "prefill_chunked_tokens": self.prefill_chunked_tokens,
            "recomputed_tokens_total": self.scheduler.recomputed_tokens_total,
            "recompute_ms_total": self.scheduler.recompute_ns_total / 1_000_000,
            "int8_spill_requests_live": len(self._int8_preemption_spills),
            "int8_spill_bytes_live": spill_bytes_live,
            "int8_spill_events_total": self._int8_spill_events,
            "int8_spill_bytes_total": self._int8_spill_bytes_total,
            "int8_spill_write_ms_total": self._int8_spill_write_ms_total,
            "int8_spill_restore_ms_total": self._int8_spill_restore_ms_total,
        }

    def recompute_report(self) -> dict[str, object]:
        """Aggregate recompute cost, including requests still in flight.

        Terminal requests are banked into the scheduler totals as they end; active and
        waiting ones are added live so a soak can sample this at any moment.
        """
        in_flight_tokens = 0
        in_flight_ns = 0
        in_flight_preemptions = 0
        for request in list(self.scheduler.active.values()) + list(self.scheduler.waiting):
            in_flight_tokens += request.recomputed_token_count
            in_flight_ns += request.recompute_ns
            in_flight_preemptions += request.preempted_count
        return {
            "preemptions": self.scheduler.preemption_count,
            "in_flight_preemptions": in_flight_preemptions,
            "progress_epoch": self.scheduler.progress_epoch,
            "recomputed_tokens": self.scheduler.recomputed_tokens_total + in_flight_tokens,
            "recompute_ms": (self.scheduler.recompute_ns_total + in_flight_ns) / 1_000_000,
            "int8_spill_requests_live": len(self._int8_preemption_spills),
            "int8_spill_events": self._int8_spill_events,
            "int8_spill_bytes": self._int8_spill_bytes_total,
            "int8_spill_write_ms": self._int8_spill_write_ms_total,
            "int8_spill_restore_ms": self._int8_spill_restore_ms_total,
        }

    def _publish_prefix(
        self, request: GenerationRequest, next_token_id: int | None,
        token_ids: list[int] | None = None,
    ) -> None:
        sequence = request.prompt_token_ids if token_ids is None else token_ids
        if request.allocation is not None and sequence:
            # An exact entry caches a token decision, not the logits which produced it.
            # Only parameter-free greedy requests may publish that decision.  Sampled
            # requests still publish reusable KV blocks, but must rerun the final prompt
            # position so their private RNG stream advances exactly once per token.
            cached_prediction = (
                next_token_id
                if request.sampling.can_reuse_cached_prediction
                else None
            )
            self.prefix_cache.publish(sequence, request.allocation, cached_prediction)

    def _finish_request(self, request: GenerationRequest, reason: str) -> None:
        self.scheduler.finish(request.request_id, reason=reason)
        self.sampler.forget([request.request_id])

    def _fail_request(self, request: GenerationRequest, reason: str) -> None:
        self.scheduler.fail(request.request_id, reason)
        self.sampler.forget([request.request_id])

    def _complete_prefill(self, request: GenerationRequest, predicted_token: int | None) -> None:
        """Move a fully prefilled request into decode, handling resumption after preemption."""
        if request.resuming:
            # The KV now covers prompt + generated[:-1]; the last generated token is the
            # pending decode input. The prefill's own prediction is discarded so a resumed
            # request continues exactly where it was preempted. Only the original prompt
            # is (re)published: generated continuations are not useful prefixes.
            request.next_token_id = request.output_token_ids[-1]
            self._publish_prefix(request, None, request.prompt_token_ids)
            self.scheduler.mark_decoding(request.request_id)
            request.complete_resumption()
            return
        if predicted_token is None:
            self._fail_request(request, "INVALID_PREFIX_ENTRY")
            return
        request.next_token_id = int(predicted_token)
        self._publish_prefix(request, request.next_token_id)
        self.scheduler.mark_decoding(request.request_id)
        request.append_token(request.next_token_id)
        reason = self._finish_reason(request, request.next_token_id)
        if reason is not None:
            self._finish_request(request, reason)

    # ------------------------------------------------------------------
    # D1: prefill a sequence, store its (rotated) K,V into the pool
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, request: GenerationRequest) -> None:
        """Compatibility wrapper supporting both fresh and prefix-hit requests."""
        self.prefill_chunks([(request, request.remaining_prefill_tokens)])

    @torch.inference_mode()
    def prefill_batch(self, requests: list[GenerationRequest]) -> None:
        """Prefill newly admitted requests in one padded, masked model forward."""
        from engine.cache.pool_cache import BatchedPoolBackedPrefillCache

        if not requests:
            return
        if len(requests) > self.max_active:
            raise ValueError("prefill batch exceeds max_active")

        # Stock SDPA computes prefill attention; the cache adapter writes the already
        # RoPE-rotated K/V directly into this engine's authoritative shared pool.
        self.model.config._attn_implementation = "sdpa"
        if hasattr(self.model.config, "_attn_implementation_internal"):
            self.model.config._attn_implementation_internal = "sdpa"

        for request in requests:
            if request.state is not RequestState.PREFILLING or request.allocation is None:
                raise RuntimeError("every request must be admitted before prefill")
            if not request.prompt_token_ids:
                raise ValueError("prefill requires prompt_token_ids")
            if request.prefilled_token_count:
                raise ValueError("partially-prefilled requests must use prefill_chunks")

        viable = []
        for request in requests:
            if request.state is not RequestState.PREFILLING:
                continue  # preempted while making room for an earlier request
            if self._capacity_or_fail(request, request.prefill_token_count):
                viable.append(request)
        requests = [
            request for request in viable
            if request.state is RequestState.PREFILLING and request.allocation is not None
        ]
        if not requests:
            return

        sequences = [request.prefill_token_ids for request in requests]
        lengths_list = [len(sequence) for sequence in sequences]
        padded_length = max(lengths_list)
        max_blocks = max(len(request.block_table) for request in requests)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = next(iter(self.eos_ids), 0)
        padded_ids = [
            sequence + [pad_token_id] * (padded_length - length)
            for sequence, length in zip(sequences, lengths_list)
        ]
        masks = [
            [1] * length + [0] * (padded_length - length) for length in lengths_list
        ]
        padded_tables = [
            request.block_table + [-1] * (max_blocks - len(request.block_table))
            for request in requests
        ]
        ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        block_tables = torch.tensor(padded_tables, dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor(lengths_list, dtype=torch.int32, device=self.device)
        cache = BatchedPoolBackedPrefillCache(
            self.key_pool, self.value_pool, block_tables, seq_lens, padded_length,
            self.key_scale_pool, self.value_scale_pool, key_smoothing=self.key_smoothing,
        )
        timer = self._gpu_timer()
        hidden = self._decoder()(
            input_ids=ids, attention_mask=attention_mask,
            past_key_values=cache, use_cache=True, return_dict=True,
        ).last_hidden_state
        # Only each row's last prompt token produces a generated token. Projecting the
        # whole padded chunk onto the vocabulary is ~20% of a prefill step's FLOPs for
        # 1/padded_length of the output, so the projection is applied to the gathered
        # last hidden states alone.
        rows = torch.arange(len(requests), device=self.device)
        last_positions = seq_lens.to(dtype=torch.long) - 1
        next_tokens = self._sample(self._lm_head()(hidden[rows, last_positions]), requests)
        self._gpu_elapsed(timer, "prefill_gpu_ms")

        for request, token in zip(requests, next_tokens):
            if not self.block_manager.append_tokens(
                request.request_id, request.remaining_prefill_tokens
            ):
                raise RuntimeError("prefill capacity was acquired but could not be committed")
            request.advance_prefill(request.remaining_prefill_tokens)
            self._complete_prefill(request, int(token))

    @torch.inference_mode()
    def prefill_chunks(
        self, plans: list[tuple[GenerationRequest, int]]
    ) -> None:
        """Run one causal paged-prefill chunk for each planned request."""
        if not plans:
            return
        if len(plans) > self.max_active:
            raise ValueError("prefill chunk batch exceeds max_active")
        for request, count in plans:
            if request.state is not RequestState.PREFILLING or request.allocation is None:
                raise RuntimeError("every chunk request must be admitted and PREFILLING")
            if not request.prompt_token_ids or not 0 < count <= request.remaining_prefill_tokens:
                raise ValueError("invalid prefill chunk plan")

        # Keep the established SDPA fast path for a batch of complete fresh prompts.
        # It is substantially better for short prompts; chunk attention is selected only
        # when a request really needs resumable prefill.
        if all(
            request.prefilled_token_count == 0 and count == request.prefill_token_count
            for request, count in plans
        ):
            self.prefill_sdpa_calls += 1
            self.prefill_sdpa_tokens += sum(count for _, count in plans)
            self.last_step_prefill_path = "sdpa"
            self.prefill_batch([request for request, _ in plans])
            return
        self.prefill_chunked_calls += 1
        self.prefill_chunked_tokens += sum(count for _, count in plans)
        self.last_step_prefill_path = "chunked"

        plans = self._prefill_viable(plans)
        if not plans:
            return

        self._set_attention(self.PREFILL_ATTN_NAME)

        count = len(plans)
        row_bucket = self._prefill_row_bucket(count)
        use_graph = (
            self.prefill_cuda_graphs and row_bucket is not None
            and self.prefill_attention not in self._prefill_graph_unsupported
        )
        row_count = row_bucket if use_graph else count
        total_len = max(request.prefilled_token_count + n for request, n in plans)
        # Graphs are shape-fixed: the chunk width is the full chunk and, for the SDPA
        # path, the gathered prefix length is rounded up to a bucket. Eager runs use the
        # exact sizes.
        if use_graph:
            width = self.prefill_chunk_size
            context_len = _prefill_context_bucket(total_len, capacity=self._pool_token_capacity) if self.prefill_attention == "sdpa" else 0
            key = (row_count, self.prefill_attention, context_len)
            graph = self._prefill_graphs.get(key)
            if graph is None:
                # Capture stages inert rows of its own; the real batch is staged after.
                self.lazy_graph_captures += 1
                graph = self._capture_prefill_graph(row_count, context_len)
            if graph is None:  # capture failed for this kind: fall back to eager
                use_graph = False
                row_count = count
        if not use_graph:
            width = max(n for _, n in plans)
            context_len = total_len
        self._prepare_prefill_metadata(plans, row_count)
        self._set_prefill_context(row_count, context_len if context_len else total_len)
        timer = self._gpu_timer()
        try:
            if use_graph:
                logits = graph.replay()
            else:
                logits = self._prefill_forward(row_count, width)
            # One device-to-host transfer for the batch.
            tokens = self._sample(logits[:count], [request for request, _ in plans])
        finally:
            _clear_prefill_ctx()

        self._commit_prefill(plans, tokens)
        self._gpu_elapsed(timer, "prefill_gpu_ms")

    def _decoder(self):
        """The transformer body without the vocabulary projection."""
        decoder = getattr(self.model, "model", None)
        if decoder is None and hasattr(self.model, "get_decoder"):
            decoder = self.model.get_decoder()
        if decoder is None:
            raise RuntimeError("model does not expose its decoder body")
        return decoder

    def _lm_head(self):
        head = getattr(self.model, "lm_head", None)
        if head is None and hasattr(self.model, "get_output_embeddings"):
            head = self.model.get_output_embeddings()
        if head is None:
            raise RuntimeError("model does not expose its output projection")
        return head

    def _prepare_prefill_metadata(
        self, plans: list[tuple[GenerationRequest, int]], row_count: int,
    ) -> None:
        """Stage one chunk batch into the persistent prefill buffers.

        Rows `len(plans)..row_count` and every column past a row's chunk are inert:
        pad token, chunk length 0, position 0, block table -1. The write kernels store
        nothing for them and the attention kernels visit no keys, so a captured graph
        can be replayed on a smaller batch, and capture itself can run on no batch at all.
        """
        chunk = self.prefill_chunk_size
        if row_count > self.max_active or len(plans) > row_count:
            raise ValueError("invalid prefill row count")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = next(iter(self.eos_ids), 0)
        ids_rows, position_rows, starts, counts = [], [], [], []
        for request, n in plans:
            if n > chunk:
                raise ValueError("prefill chunk exceeds the engine's chunk size")
            start = request.prefilled_token_count
            tokens = request.prefill_token_ids[start:start + n]
            ids_rows.append(tokens + [pad_token_id] * (chunk - n))
            # Padded positions are never observed, but valid absolute positions keep RoPE
            # consistent with later decode steps; inert rows sit at position 0.
            position_rows.append(list(range(start, start + n)) + [start] * (chunk - n))
            starts.append(start)
            counts.append(n)
        for _ in range(row_count - len(plans)):
            ids_rows.append([pad_token_id] * chunk)
            position_rows.append([0] * chunk)
            starts.append(0)
            counts.append(0)
        self._prefill_host_input_ids[:row_count] = torch.tensor(ids_rows, dtype=torch.long)
        self._prefill_host_position_ids[:row_count] = torch.tensor(position_rows, dtype=torch.long)
        self._prefill_host_starts[:row_count] = torch.tensor(starts, dtype=torch.int32)
        self._prefill_host_chunk_lens[:row_count] = torch.tensor(counts, dtype=torch.int32)
        tables = self._prefill_host_block_tables
        for row in range(row_count):
            if row < len(plans):
                table = plans[row][0].block_table
                tables[row, :len(table)] = torch.tensor(table, dtype=torch.int32)
                tables[row, len(table):] = -1
            else:
                tables[row].fill_(-1)
        self._prefill_device_input_ids[:row_count].copy_(
            self._prefill_host_input_ids[:row_count], non_blocking=True)
        self._prefill_device_position_ids[:row_count].copy_(
            self._prefill_host_position_ids[:row_count], non_blocking=True)
        self._prefill_device_starts[:row_count].copy_(
            self._prefill_host_starts[:row_count], non_blocking=True)
        self._prefill_device_chunk_lens[:row_count].copy_(
            self._prefill_host_chunk_lens[:row_count], non_blocking=True)
        self._prefill_device_block_tables[:row_count].copy_(
            self._prefill_host_block_tables[:row_count], non_blocking=True)

    def _prepare_packed_metadata(
        self, plans: list[tuple[GenerationRequest, int]], row_count: int,
    ) -> None:
        """Stage the packed layout of one chunk batch: tokens back to back, then padding.

        The padded attention layout keeps the staged chunk width; every padded slot that
        holds no real token points at packed index 0, which attention ignores (its chunk
        length excludes it and the KV writers skip it). Packed slots past the real tokens
        point at padded slot 0 and their outputs are never read.
        """
        chunk = self.prefill_chunk_size
        total = sum(count for _, count in plans)
        packed_width = self._active_packed_width
        if total > packed_width:
            raise ValueError("packed chunk tokens exceed the packed width")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = next(iter(self.eos_ids), 0)
        ids, positions, to_packed, last = [], [], [], []
        to_padded = [0] * (row_count * chunk)
        for row, (request, count) in enumerate(plans):
            start = request.prefilled_token_count
            base = len(ids)
            ids.extend(request.prefill_token_ids[start:start + count])
            positions.extend(range(start, start + count))
            slots = range(row * chunk, row * chunk + count)
            to_padded[row * chunk:row * chunk + count] = range(base, base + count)
            to_packed.extend(slots)
            last.append(base + count - 1)
        padding = packed_width - total
        ids.extend([pad_token_id] * padding)
        positions.extend([0] * padding)
        to_packed.extend([0] * padding)
        last.extend([0] * (row_count - len(plans)))
        width = row_count * chunk
        self._packed_host_ids[:packed_width] = torch.tensor(ids, dtype=torch.long)
        self._packed_host_positions[:packed_width] = torch.tensor(positions, dtype=torch.long)
        self._packed_host_to_packed[:packed_width] = torch.tensor(to_packed, dtype=torch.long)
        self._packed_host_to_padded[:width] = torch.tensor(to_padded, dtype=torch.long)
        self._packed_host_last[:row_count] = torch.tensor(last, dtype=torch.long)
        self._packed_device_ids[:packed_width].copy_(self._packed_host_ids[:packed_width], non_blocking=True)
        self._packed_device_positions[:packed_width].copy_(self._packed_host_positions[:packed_width], non_blocking=True)
        self._packed_device_to_packed[:packed_width].copy_(self._packed_host_to_packed[:packed_width], non_blocking=True)
        self._packed_device_to_padded[:width].copy_(self._packed_host_to_padded[:width], non_blocking=True)
        self._packed_device_last[:row_count].copy_(self._packed_host_last[:row_count], non_blocking=True)

    def _set_prefill_context(self, row_count: int, total_len: int) -> None:
        attention_cache: dict = {}
        if self.prefill_attention in {"flash", "flash_dense"}:
            # These lengths already exist in pinned host staging. Group them once per
            # engine step, not once per transformer layer; reading the CUDA tensor in
            # `flash_paged_prefill` caused 28 device synchronizations on Qwen3.
            lengths = self._prefill_host_chunk_lens[:row_count].tolist()
            grouped: dict[int, list[int]] = {}
            for row, length in enumerate(lengths):
                if length > 0:
                    grouped.setdefault(int(length), []).append(row)
            if self.prefill_attention == "flash_dense":
                starts = self._prefill_host_starts[:row_count].tolist()
                dense_grouped: dict[tuple[int, int], list[int]] = {}
                for row, length in enumerate(lengths):
                    if length > 0:
                        dense_grouped.setdefault((int(length), int(starts[row] + length)), []).append(row)
                if (len(dense_grouped) == 1
                        and next(iter(dense_grouped.values())) == list(range(row_count))):
                    length, total = next(iter(dense_grouped))
                    attention_cache["flash_dense_row_groups"] = ((length, total, None),)
                else:
                    attention_cache["flash_dense_row_groups"] = tuple(
                        (length, total, torch.tensor(rows, device=self.device, dtype=torch.long))
                        for (length, total), rows in dense_grouped.items()
                    )
            elif len(grouped) == 1 and next(iter(grouped.values())) == list(range(row_count)):
                length = next(iter(grouped))
                attention_cache["flash_row_groups"] = ((length, None),)
            else:
                attention_cache["flash_row_groups"] = tuple(
                    (length, torch.tensor(rows, device=self.device, dtype=torch.long))
                    for length, rows in grouped.items()
                )
        _set_prefill_ctx(_PrefillContext(
            self.key_pool, self.value_pool,
            self._prefill_device_block_tables[:row_count],
            self._prefill_device_starts[:row_count],
            self._prefill_device_chunk_lens[:row_count],
            self.key_scale_pool, self.value_scale_pool,
            attention=self.prefill_attention,
            backend=self.prefill_backend,
            total_len=total_len,
            prefill_block_m=self.prefill_block_m,
            prefill_block_n=self.prefill_block_n,
            key_smoothing=self.key_smoothing,
            int8_prefill_attention=self.int8_prefill_attention,
            sdpa_cache=attention_cache,
        ))

    def _prefill_forward(self, row_count: int, width: int) -> torch.Tensor:
        """One chunked prefill forward over the staged buffers -> `[rows, vocab]` logits.

        The vocabulary projection runs on each row's last chunk token only; a row that
        does not finish its prompt this step simply discards the result. Logits rather
        than tokens, because sampling is per request and cannot live inside a graph
        captured for a shape rather than for a set of requests.
        """
        input_ids = self._prefill_device_input_ids[:row_count, :width]
        position_ids = self._prefill_device_position_ids[:row_count, :width]
        if width != self.prefill_chunk_size:
            input_ids = input_ids.contiguous()
            position_ids = position_ids.contiguous()
        chunk_lens = self._prefill_device_chunk_lens[:row_count]
        hidden = self._decoder()(
            input_ids=input_ids, position_ids=position_ids, use_cache=False, return_dict=True,
        ).last_hidden_state
        last = (chunk_lens.to(torch.long) - 1).clamp_(min=0)
        rows = torch.arange(row_count, device=hidden.device)
        return self._write_logits(self._lm_head()(hidden[rows, last]))

    def _write_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Copy a forward's logits into the shared buffer and return that view.

        Under capture the copy is recorded, so every graph writes its result to the same
        address and replay leaves it there for the sampler.
        """
        if self._logits_buffer is None:
            return logits
        target = self._logits_buffer[:logits.shape[0]]
        target.copy_(logits)
        return target

    def _warmup_context_buckets(self) -> list[int]:
        """Every gathered-context bucket a chunk batch can reach on this engine.

        A chunk's context is its prompt so far - or prompt plus generated tokens for a
        request re-prefilling after preemption - bounded by the model context and by the
        pool. A fixed list ending at 2048 left the long profile capturing 4096-context
        graphs inside the timed window (journal, Phase 2b): p999 rose while p99 fell.
        """
        capacity = self.key_pool[0].shape[0] * self.block_size
        if self.max_model_len:
            capacity = min(capacity, int(self.max_model_len))
        buckets = []
        bucket = 256
        while True:
            buckets.append(min(bucket, self._pool_token_capacity))
            if bucket >= capacity:
                break
            bucket *= 2
        return sorted(set(buckets))

    def _capture_prefill_graph(self, row_count: int, context_len: int):
        """Capture one prefill graph, or record that this attention kind cannot be captured."""
        from engine.graphs.paged_prefill_graph import capture_paged_prefill_graph

        if self.prefill_attention in self._prefill_graph_unsupported:
            return None
        if self._prefill_graph_pool is None:
            self._prefill_graph_pool = torch.cuda.graph_pool_handle()
        try:
            graph = capture_paged_prefill_graph(
                self, rows=row_count, context_len=context_len, pool=self._prefill_graph_pool,
            )
        except RuntimeError as error:
            # A kernel that is not capture-safe surfaces here (typically a synchronizing
            # op inside the forward). Serving continues eagerly; the reason is kept so a
            # benchmark can report it instead of silently measuring the eager path.
            self._prefill_graph_unsupported[self.prefill_attention] = str(error).splitlines()[0][:200]
            torch.cuda.synchronize()
            return None
        self._prefill_graphs[(row_count, self.prefill_attention, context_len)] = graph
        return graph

    def _plan_prefill_chunks(self) -> list[tuple[GenerationRequest, int]]:
        if self.idle_prefill_tokens:
            prefilling = [
                request for request in self.scheduler.active.values()
                if request.state is RequestState.PREFILLING
            ]
            decoding = any(
                request.state is RequestState.DECODING for request in self.scheduler.active.values()
            )
            # One fresh prompt on an otherwise idle engine goes through the single-forward
            # fast path whole. A burst of several prompts keeps chunking: interleaved
            # chunks gave each of them an earlier first token than serial whole prefills
            # (1.7B chat, four closed-loop clients: TTFT +7% with whole prefills).
            if (len(prefilling) == 1 and not decoding
                    and prefilling[0].prefilled_token_count == 0
                    and prefilling[0].prefill_token_count <= self.idle_prefill_tokens):
                return [(prefilling[0], prefilling[0].prefill_token_count)]
        return self.scheduler.plan_prefill(
            chunk_size=self.prefill_chunk_size,
            token_budget=self.max_prefill_tokens_per_iteration,
        )

    def cancel(self, request_id: str, reason: str = "CANCELLED_BY_CLIENT") -> GenerationRequest:
        """Cancel queued, partially-prefilled, or decoding work and release its KV."""
        request = self.scheduler.cancel(request_id, reason=reason)
        self._int8_preemption_spills.pop(request_id, None)
        self.sampler.forget([request.request_id])
        return request

    def submit(self, request: GenerationRequest) -> bool:
        """Submit an externally-created request to the bounded online scheduler."""
        return self.scheduler.submit(request)

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.active)

    @torch.inference_mode()
    def step(self) -> None:
        """Run one decode-first scheduling iteration under the prefill budget.

        A step that also prefills costs every decoding sequence a longer gap between its
        tokens, because the prefill forward runs in the same iteration. The counters and
        `last_step_prefill_tokens` let a benchmark separate those gaps from pure decode
        gaps instead of inferring the split from a skewed distribution.
        """
        self.last_step_timing = {}
        decoding = [
            request for request in self.scheduler.active.values()
            if request.state is RequestState.DECODING
        ]
        fused = self.fused_step and bool(decoding)
        rows: list[GenerationRequest] = []
        if fused:
            # Capacity for the decode rows is taken before admission, as the separate
            # decode forward would have; the forward itself waits for the prefill plan so
            # both can share it.
            rows = self._decode_viable(decoding)
        elif decoding:
            self.decode_step(decoding)
        admitted = self.scheduler.admit_available(max_active_requests=self.max_active)
        for request in admitted:
            if self._restore_int8_spill(request):
                continue
            if request.remaining_prefill_tokens == 0:
                self._complete_prefill(request, request.cached_next_token_id)
        plans = self._plan_prefill_chunks()
        self.last_step_prefill_tokens = sum(count for _, count in plans)
        self.last_step_decode_rows = len(decoding)
        self.last_step_prefill_path = ""
        if fused:
            if self._fused_rows_step(rows, plans):
                self.fused_steps += 1
        elif plans:
            self.prefill_chunks(plans)
        if plans:
            self.prefill_steps += 1
        elif decoding:
            self.decode_only_steps += 1

    def _set_attention(self, name: str) -> None:
        config = self.model.config
        config._attn_implementation = name
        if hasattr(config, "_attn_implementation_internal"):
            config._attn_implementation_internal = name

    def _graph_bucket(self, count: int) -> int | None:
        return next(
            (size for size in self.cuda_graph_batch_sizes if count <= size <= self.max_active),
            None,
        )

    # Chunk batches are usually one or two rows under the 128-token budget, and every
    # padded row costs a full chunk of per-token work, so the row buckets start at 1
    # rather than at the smallest decode bucket. Above this many rows the fused graph
    # would spend more on inert rows than a second forward costs; eager runs exact.
    FUSED_PREFILL_ROW_LIMIT = 4

    def _prefill_row_bucket(self, count: int, limit: int | None = None) -> int | None:
        if not self.cuda_graph_batch_sizes:
            return None
        sizes = (1,) + self.cuda_graph_batch_sizes
        bucket = next((size for size in sizes if count <= size <= self.max_active), None)
        if bucket is not None and limit is not None and bucket > limit:
            return None
        return bucket

    def _prefill_viable(
        self, plans: list[tuple[GenerationRequest, int]],
    ) -> list[tuple[GenerationRequest, int]]:
        """Acquire KV capacity for each planned chunk, dropping preempted requests."""
        viable_plans = []
        for request, count in plans:
            if request.state is not RequestState.PREFILLING:
                continue  # preempted while making room for an earlier request
            target = request.prefilled_token_count + count
            if self._capacity_or_fail(request, target):
                viable_plans.append((request, count))
        return [
            (request, count) for request, count in viable_plans
            if request.state is RequestState.PREFILLING and request.allocation is not None
        ]

    def _commit_prefill(
        self, plans: list[tuple[GenerationRequest, int]], tokens: list[int],
    ) -> None:
        for (request, n), token in zip(plans, tokens):
            if not self.block_manager.append_tokens(request.request_id, n):
                raise RuntimeError("prefill capacity was acquired but could not be committed")
            request.advance_prefill(n)
            if request.remaining_prefill_tokens == 0:
                self._complete_prefill(request, int(token))

    def _sample(self, logits, requests: list[GenerationRequest]) -> list[int]:
        """One token per row under each request's own sampling parameters.

        `logits` is `[len(requests), vocab]`. Penalty contexts are assembled only when a
        request actually asks for penalties, so the common greedy batch costs one argmax
        and one device-to-host copy, exactly as before sampling existed.
        """
        params = [request.sampling for request in requests]
        contexts = None
        if any(setting.has_penalties for setting in params):
            contexts = [
                (request.prompt_token_ids + request.output_token_ids)
                if request.sampling.penalize_prompt else request.output_token_ids
                for request in requests
            ]
        result = self.sampler.sample(
            logits, params, contexts, [request.request_id for request in requests],
        )
        for row, entries in result.logprobs.items():
            requests[row].output_logprobs.append(entries)
        return result.token_ids

    def _finish_reason(self, request: GenerationRequest, token: int) -> str | None:
        """Why this token ends the request, or None if it does not.

        Per-request stop tokens come first: a caller that asked to stop on a token gets
        `STOP` even if that token is also the model's EOS. `ignore_eos` lets a benchmark
        or a continuation request run to its length bound.
        """
        sampling = request.sampling
        if token in sampling.stop_token_ids:
            return "STOP"
        if not sampling.ignore_eos and token in self.eos_ids:
            return "EOS"
        if len(request.output_token_ids) >= request.max_new_tokens:
            return "LENGTH"
        return None

    def _live_rows(self, decode_count: int, decode_rows: int, prefill_count: int):
        """Index of the live rows inside a fused step's `[decode_rows + prefill_rows]`
        logits buffer: the real decode rows, then the real chunk rows, skipping the
        padding the graph's shape requires. Cached because it is a host-to-device copy
        on the step's critical path."""
        key = (decode_count, decode_rows, prefill_count)
        index = self._live_row_index.get(key)
        if index is None:
            rows = list(range(decode_count)) + [
                decode_rows + row for row in range(prefill_count)
            ]
            index = torch.tensor(rows, dtype=torch.long, device=self.device)
            self._live_row_index[key] = index
        return index

    def _commit_decode(self, active: list[GenerationRequest], tokens: list[int]) -> None:
        for s, token in zip(active, tokens):
            self.block_manager.append_tokens(s.request_id)
            tok = int(token)
            s.next_token_id = tok
            s.append_token(tok)
            reason = self._finish_reason(s, tok)
            if reason is not None:
                self._finish_request(s, reason)

    def _set_fused_contexts(
        self, *, decode_rows: int, block_tables, seq_lens, block_n: int, num_warps: int,
        prefill_rows: int, total_len: int, width: int, max_sequence_length: int = 0,
    ) -> None:
        _set_batch_ctx(_BatchContext(
            key_pool=self.key_pool, value_pool=self.value_pool,
            block_tables=block_tables, seq_lens=seq_lens, block_size=self.block_size,
            decode_block_n=block_n, decode_num_warps=num_warps,
            decode_num_splits=self.decode_num_splits,
            key_scale_pool=self.key_scale_pool, value_scale_pool=self.value_scale_pool,
            decode_kernel=self.decode_attention,
            decode_backend=self.decode_backend, decode_max_len=max_sequence_length,
            key_smoothing=self.key_smoothing, int8_decode_tuned=self.int8_decode_tuned,
        ))
        self._set_prefill_context(prefill_rows, total_len)
        if self.packed_fused_prefill:
            _set_fused_ctx(_FusedContext(
                decode_rows, prefill_rows, width, packed_tokens=self._active_packed_width,
                to_padded=self._packed_device_to_padded[:prefill_rows * width],
                to_packed=self._packed_device_to_packed[:self._active_packed_width],
            ))
        else:
            _set_fused_ctx(_FusedContext(decode_rows, prefill_rows, width))

    @staticmethod
    def _clear_fused_contexts() -> None:
        _clear_fused_ctx()
        _clear_prefill_ctx()
        _clear_batch_ctx()

    def _fused_forward(self, decode_rows: int, prefill_rows: int, width: int) -> torch.Tensor:
        """One packed forward over the staged decode and prefill buffers -> logits.

        Returns `[decode_rows + prefill_rows, vocab]`: the decode rows first, then each
        chunk row at its last valid token (the caller discards rows whose prompt is not
        finished). The vocabulary projection runs on exactly those rows, and the result
        lands in the shared logits buffer so a captured graph has a fixed output address.
        """
        decode_ids = self._device_input_ids[:decode_rows].reshape(1, decode_rows)
        decode_positions = self._device_position_ids[:decode_rows].reshape(1, decode_rows)
        if self.packed_fused_prefill:
            chunk_ids = self._packed_device_ids[:self._active_packed_width].reshape(1, -1)
            chunk_positions = self._packed_device_positions[:self._active_packed_width].reshape(1, -1)
        else:
            chunk_ids = self._prefill_device_input_ids[:prefill_rows, :width].reshape(1, -1)
            chunk_positions = self._prefill_device_position_ids[:prefill_rows, :width].reshape(1, -1)
        input_ids = torch.cat((decode_ids, chunk_ids), dim=1)
        position_ids = torch.cat((decode_positions, chunk_positions), dim=1)
        hidden = self._decoder()(
            input_ids=input_ids, position_ids=position_ids, use_cache=False, return_dict=True,
        ).last_hidden_state[0]
        chunk_lens = self._prefill_device_chunk_lens[:prefill_rows]
        if self.packed_fused_prefill:
            last = decode_rows + self._packed_device_last[:prefill_rows]
        else:
            last = (
                decode_rows
                + torch.arange(prefill_rows, device=hidden.device) * width
                + (chunk_lens.to(torch.long) - 1).clamp_(min=0)
            )
        picked = torch.cat((hidden[:decode_rows], hidden[last]), dim=0)
        return self._write_logits(self._lm_head()(picked))

    def _packed_width_for(self, tokens: int) -> int:
        """Smallest captured packed width that holds `tokens` chunk tokens."""
        return next(width for width in self._packed_widths if width >= tokens)

    def _capture_fused_graph(self, key: tuple):
        """Capture one fused step graph, or record that this attention kind cannot be."""
        from engine.graphs.fused_step_graph import capture_fused_step_graph

        decode_rows, prefill_rows, kind, context_len, block_n, num_warps = key[:6]
        # Packed engines key fused graphs by packed width as well (`_packed_width_for`).
        self._active_packed_width = key[6] if len(key) > 6 else self._packed_tokens
        tag = f"fused:{kind}"
        if tag in self._prefill_graph_unsupported:
            return None
        if self._prefill_graph_pool is None:
            self._prefill_graph_pool = torch.cuda.graph_pool_handle()
        try:
            graph = capture_fused_step_graph(
                self, decode_rows=decode_rows, prefill_rows=prefill_rows,
                context_len=context_len, block_n=block_n, num_warps=num_warps,
                pool=self._prefill_graph_pool,
            )
        except RuntimeError as error:
            self._prefill_graph_unsupported[tag] = str(error).splitlines()[0][:200]
            torch.cuda.synchronize()
            return None
        self._fused_graphs[key] = graph
        return graph

    def _fused_rows_step(
        self, rows: list[GenerationRequest], plans: list[tuple[GenerationRequest, int]],
    ) -> bool:
        """Advance the decode rows and the planned chunks in one forward.

        Falls back to the separate paths when one side is empty: a decode-only step is
        the plain decode forward, and a prefill-only step keeps the fresh-prompt fast
        path. Returns True when the fused forward ran.
        """
        plans = self._prefill_viable(plans) if plans else []
        # Prefill capacity may have preempted a decode row admitted after the prefilling
        # request; such a row has no KV to write into anymore.
        rows = [r for r in rows if r.state is RequestState.DECODING and r.allocation is not None]
        if not plans:
            if rows:
                self._decode_rows(rows)
            return False
        if not rows:
            self.prefill_chunks(plans)
            return False

        host_started = perf_counter() if self.instrument else None
        self.prefill_chunked_calls += 1
        self.prefill_chunked_tokens += sum(count for _, count in plans)
        self.last_step_prefill_path = "fused"
        decode_count, prefill_count = len(rows), len(plans)
        max_sequence_length = max(r.allocation.sequence_length + 1 for r in rows)
        block_n, num_warps = select_paged_decode_config(
            max_sequence_length, decode_count, self.decode_attention,
        )
        decode_bucket = self._graph_bucket(decode_count)
        prefill_bucket = self._prefill_row_bucket(prefill_count, self.FUSED_PREFILL_ROW_LIMIT)
        total_len = max(request.prefilled_token_count + n for request, n in plans)
        use_graph = (
            self.decode_cuda_graphs and self.prefill_cuda_graphs
            and decode_bucket is not None and prefill_bucket is not None
            and f"fused:{self.prefill_attention}" not in self._prefill_graph_unsupported
        )
        graph = None
        planned_tokens = sum(count for _, count in plans)
        packed_width = self._packed_width_for(planned_tokens)
        if use_graph:
            width = self.prefill_chunk_size
            context_len = _prefill_context_bucket(total_len, capacity=self._pool_token_capacity) if self.prefill_attention == "sdpa" else 0
            key = (decode_bucket, prefill_bucket, self.prefill_attention, context_len, block_n, num_warps)
            if self.packed_fused_prefill:
                key += (packed_width,)
            graph = self._fused_graphs.get(key)
            if graph is None:
                self.lazy_graph_captures += 1
                graph = self._capture_fused_graph(key)
            use_graph = graph is not None
        if use_graph:
            decode_rows, prefill_rows = decode_bucket, prefill_bucket
        else:
            decode_rows, prefill_rows = decode_count, prefill_count
            # Packed keeps the staged width for attention's padded view; the per-token
            # modules see only the packed budget either way.
            width = self.prefill_chunk_size if self.packed_fused_prefill else max(n for _, n in plans)
            context_len = total_len
        self._set_attention(self.FUSED_ATTN_NAME)
        _, _, block_tables, seq_lens = self._prepare_decode_metadata(
            rows, graph_bucket_size=decode_rows if use_graph else None,
        )
        self._prepare_prefill_metadata(plans, prefill_rows)
        if self.packed_fused_prefill:
            self._active_packed_width = packed_width
            self._prepare_packed_metadata(plans, prefill_rows)
        self._host_elapsed(host_started, "host_stage_ms")
        timer = self._gpu_timer()
        self._set_fused_contexts(
            decode_rows=decode_rows, block_tables=block_tables, seq_lens=seq_lens,
            block_n=block_n, num_warps=num_warps, prefill_rows=prefill_rows,
            total_len=context_len if context_len else total_len, width=width,
            max_sequence_length=max_sequence_length,
        )
        try:
            if use_graph:
                logits = graph.replay()
            else:
                logits = self._fused_forward(decode_rows, prefill_rows, width)
            # One device-to-host transfer for decode and prefill together: the padded
            # rows a graph's shape requires are dropped on the device first.
            sync_started = perf_counter() if self.instrument else None
            live = logits.index_select(
                0, self._live_rows(decode_count, decode_rows, prefill_count),
            )
            tokens = self._sample(live, rows + [request for request, _ in plans])
            self._host_elapsed(sync_started, "sync_ms")
        finally:
            self._clear_fused_contexts()
        self._gpu_elapsed(timer, "fused_gpu_ms")
        # `tokens` is already compacted to the live rows: decode rows, then chunk rows.
        self._commit_decode(rows, tokens[:decode_count])
        self._commit_prefill(plans, tokens[decode_count:decode_count + prefill_count])
        return True

    # ------------------------------------------------------------------
    # D2: one batched decode step over all active sequences
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def decode_step(self, active: list[GenerationRequest]) -> None:
        """Advance all active sequences by one token via a single batched forward."""
        active = self._decode_viable(active)
        if active:
            self._decode_rows(active)

    def _decode_viable(self, active: list[GenerationRequest]) -> list[GenerationRequest]:
        """Acquire each row's next KV slot, in FCFS priority order.

        A request that cannot grow preempts newer requests (which then vanish from this
        batch) and is failed only when nothing older can help. Victims are always later
        in priority order, but a resumed request sits at the end of the admission order
        with an old arrival time, so re-check at the end.
        """
        viable = []
        for s in sorted(active, key=lambda item: (item.created_ns, item.request_id)):
            if s.state is not RequestState.DECODING or s.allocation is None:
                continue  # preempted earlier in this loop
            if self._capacity_or_fail(s, s.allocation.sequence_length + 1):
                viable.append(s)
        return [s for s in viable if s.state is RequestState.DECODING and s.allocation is not None]

    def _decode_rows(self, active: list[GenerationRequest]) -> None:
        """Stage, run and commit one decode forward over rows that already have capacity."""
        host_started = perf_counter() if self.instrument else None
        self._set_attention(self.ATTN_NAME)
        max_sequence_length = max(
            request.allocation.sequence_length + 1 for request in active
        )
        decode_block_n, decode_num_warps = select_paged_decode_config(
            max_sequence_length, len(active), self.decode_attention,
        )
        graph_bucket_size = next(
            (
                size for size in self.cuda_graph_batch_sizes
                if len(active) <= size <= self.max_active
            ),
            None,
        ) if self.decode_cuda_graphs else None
        input_ids, position_ids, block_tables, seq_lens = self._prepare_decode_metadata(
            active, graph_bucket_size=graph_bucket_size,
        )
        self._host_elapsed(host_started, "host_stage_ms")
        timer = self._gpu_timer()

        # Stash context for the attention fn
        context = _BatchContext(
            key_pool=self.key_pool, value_pool=self.value_pool,
            block_tables=block_tables, seq_lens=seq_lens, block_size=self.block_size,
            decode_block_n=decode_block_n, decode_num_warps=decode_num_warps,
            decode_num_splits=self.decode_num_splits,
            key_scale_pool=self.key_scale_pool, value_scale_pool=self.value_scale_pool,
            decode_kernel=self.decode_attention,
            decode_backend=self.decode_backend, decode_max_len=max_sequence_length,
            key_smoothing=self.key_smoothing, int8_decode_tuned=self.int8_decode_tuned,
        )
        # Split-K's split length is a launch-time scalar baked into a CUDA graph.
        # Bucket it separately from the tile regime; otherwise a graph captured
        # at a short context can silently skip keys when replayed at a longer one.
        graph_max_len = (
            ((max_sequence_length + 127) // 128) * 128
            if self.decode_attention == "split_k" else max_sequence_length
        )
        graph_key = (graph_bucket_size, decode_block_n, decode_num_warps)
        if self.decode_attention == "split_k":
            graph_key += (graph_max_len,)
        use_graph = graph_bucket_size is not None
        if use_graph:
            graph = self._decode_graphs.get(graph_key)
            if graph is None:
                from engine.graphs import capture_paged_decode_graph
                self.lazy_graph_captures += 1
                graph = capture_paged_decode_graph(
                    self, batch_size=graph_bucket_size, block_n=decode_block_n,
                    num_warps=decode_num_warps, max_sequence_length=graph_max_len,
                )
                self._decode_graphs[graph_key] = graph
            logits = graph.replay()
        else:
            _set_batch_ctx(context)
            try:
                logits = self.model(
                    input_ids=input_ids, position_ids=position_ids,
                    use_cache=False, return_dict=True,
                ).logits
            finally:
                _clear_batch_ctx()

        # Sample next token per sequence, advance state
        # One device-to-host synchronization for the complete batch. Calling `.item()`
        # per row serializes N scalar copies and N Python-visible CUDA waits.
        sync_started = perf_counter() if self.instrument else None
        next_tokens = self._sample(logits[:len(active), -1, :], active)
        self._host_elapsed(sync_started, "sync_ms")
        self._gpu_elapsed(timer, "decode_gpu_ms")
        self._commit_decode(active, next_tokens)

    # ------------------------------------------------------------------
    # D3: the continuous loop
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self, prompts: list[str], max_new_tokens: int = 32,
        sampling: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[list[int]]:
        """Run all prompts through continuous batching. Returns output token ids per prompt.

        `sampling` is one setting for every prompt, or one per prompt; the default is
        greedy, which is what the correctness gates compare against stock Transformers.
        """
        if isinstance(sampling, list) and len(sampling) != len(prompts):
            raise ValueError("one SamplingParams per prompt is required")
        requests = []
        for i, p in enumerate(prompts):
            ids = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            setting = sampling[i] if isinstance(sampling, list) else sampling
            request = GenerationRequest(
                request_id=f"seq{i}",
                prompt_token_count=len(ids),
                max_new_tokens=max_new_tokens,
                prompt_token_ids=ids,
                sampling=setting or GREEDY,
            )
            requests.append(request)
            self.submit(request)

        while self.has_unfinished_requests:
            self.step()

        return [request.output_token_ids for request in requests]

    @torch.inference_mode()
    def warmup(self) -> dict[str, int]:
        """Pay every first-use cost before serving: Triton JIT and CUDA-Graph capture.

        Runs synthetic requests through the ordinary step loop so that every graph
        bucket is captured in both decode kernel regimes (context below and above the
        128-token boundary in `select_paged_decode_config`), and both prefill paths -
        SDPA for a fresh whole prompt, chunked for a prompt longer than one chunk - have
        compiled. Without this, each of those costs lands on the first live requests
        that need it: a capture is two eager forwards plus a device sync, and a bucket is
        first reached at exactly the load level that fills it.

        Allocator, prefix cache, scheduler and step counters are reset afterwards, so
        warmup leaves nothing behind except captured graphs and kernel caches.
        """
        if torch.device(self.device).type != "cuda":
            return {"rounds": 0, "graphs": len(self._decode_graphs)}
        vocab_size = int(getattr(self.model.config, "vocab_size", 0)) or 1000
        generator = torch.Generator().manual_seed(0)
        # Below the regime boundary even after decoding, and within one prefill chunk.
        short_prompt = max(1, min(64, self.prefill_chunk_size, self.max_prefill_tokens_per_iteration))
        # Past the boundary once prefilled, and longer than one chunk or one budget,
        # whichever is smaller, so the resumable chunk path is the one that runs.
        long_prompt = max(130, min(self.prefill_chunk_size, self.max_prefill_tokens_per_iteration) + 2)
        widths = list(self.cuda_graph_batch_sizes) or [1]
        budget = self.max_prefill_tokens_per_iteration
        # Random token ids can decode to EOS; ignore it so every round reaches decode.
        eos_ids, self.eos_ids = self.eos_ids, set()
        rounds = 0
        try:
            for width in widths:
                for length in (short_prompt, long_prompt):
                    # Prompts are admitted a budget's worth per step, so the earliest
                    # request must keep decoding until the last one has joined the batch,
                    # or the round never reaches `width` rows and that bucket is never
                    # captured in this regime.
                    steps_to_admit_all = -(-(width * length) // budget) + width
                    max_new_tokens = 3 + steps_to_admit_all
                    for index in range(width):
                        ids = torch.randint(1, vocab_size, (length,), generator=generator).tolist()
                        self.submit(GenerationRequest(
                            request_id=f"__warmup_{rounds}_{index}",
                            prompt_token_count=length, max_new_tokens=max_new_tokens,
                            prompt_token_ids=ids,
                        ))
                    steps = 0
                    while self.has_unfinished_requests and steps < 10_000:
                        self.step()
                        steps += 1
                    rounds += 1
        finally:
            self.eos_ids = eos_ids
        # Prefill graphs are captured on inert rows, so every bucket can be taken directly
        # rather than hoping the rounds above produced a chunk batch of each width.
        if (self.prefill_cuda_graphs and self.cuda_graph_batch_sizes
                and torch.device(self.device).type == "cuda"):
            contexts = [0]
            if self.prefill_attention == "sdpa":
                contexts = self._warmup_context_buckets()
            for rows in (1,) + self.cuda_graph_batch_sizes:
                for context_len in contexts:
                    key = (rows, self.prefill_attention, context_len)
                    if key not in self._prefill_graphs:
                        if self._capture_prefill_graph(rows, context_len) is None:
                            break
            # Fused step graphs for every shape the fused path can replay: each decode
            # bucket x each chunk-row bucket up to the limit x the same context buckets
            # as the prefill graphs x both decode kernel regimes. The first Kaggle run
            # covered only {1, 2} rows and contexts up to 1024, and the long profile
            # (2048-token contexts) then captured a dozen graphs inside the timed window:
            # ITL p99 doubled (64.7 -> 126.9 ms) while p50 fell 17%. A capture is two
            # eager forwards plus syncs in a live step; it belongs in warmup or nowhere.
            if self.fused_step and self.decode_cuda_graphs:
                fused_rows = [
                    rows for rows in (1,) + self.cuda_graph_batch_sizes
                    if self._prefill_row_bucket(rows, self.FUSED_PREFILL_ROW_LIMIT) == rows
                ]
                for decode_rows in self.cuda_graph_batch_sizes:
                    # A device policy may depend on both context and graph width (the Ada
                    # paired-GQA crossover does). Capture every regime that this bucket
                    # can select so the first live request never pays for it.
                    regimes = sorted({
                        select_paged_decode_config(length, decode_rows, self.decode_attention)
                        for length in (64, 128, 256, 512, 1024)
                    })
                    for prefill_rows in fused_rows:
                        for context_len in contexts:
                            for block_n, num_warps in regimes:
                                base_key = (decode_rows, prefill_rows, self.prefill_attention,
                                            context_len, block_n, num_warps)
                                keys = ([base_key + (w,) for w in self._packed_widths]
                                        if self.packed_fused_prefill else [base_key])
                                for key in keys:
                                    if key not in self._fused_graphs:
                                        self._capture_fused_graph(key)
        summary = {
            "rounds": rounds,
            "graphs": len(self._decode_graphs),
            "prefill_graphs": len(self._prefill_graphs),
            "fused_graphs": len(self._fused_graphs),
            "prefill_graph_unsupported": dict(self._prefill_graph_unsupported),
            "prefill_graph_eager_reason": self.prefill_graph_eager_reason,
            "decode_graph_eager_reason": self.decode_graph_eager_reason,
            "prefill_sdpa_calls": self.prefill_sdpa_calls,
            "prefill_chunked_calls": self.prefill_chunked_calls,
        }
        self.reset()
        self.prefill_steps = self.decode_only_steps = self.fused_steps = 0
        self.lazy_graph_captures = 0   # captures during warmup are the point of warmup
        self.last_step_prefill_tokens = self.last_step_decode_rows = 0
        self.prefill_sdpa_calls = self.prefill_chunked_calls = 0
        self.prefill_sdpa_tokens = self.prefill_chunked_tokens = 0
        self.last_step_prefill_path = ""
        return summary

    def attn_call_count(self) -> int:
        return _ATTN_CALLS
