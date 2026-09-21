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
from engine.kernels.paged_prefill import paged_prefill
from engine.kernels.sdpa_prefill import sdpa_paged_prefill
from engine.kernels.tiled_paged_prefill import tiled_paged_prefill

# Chunked-prefill attention implementations, by name:
#   sdpa       (default) gather the prefix pages into a dense tensor and call torch SDPA -
#              the same kernel the fresh-prompt fast path uses; reaches the tensor cores
#              on sm_75. Measured -40%/-66% prefill step vs per_token on the T4 and
#              token-identical to stock Transformers on the identity prompts.
#   per_token  Triton, one program per (row, head, query token); lean, bandwidth-bound on
#              redundant prefix reads. The previous default and the measured baseline.
#   tiled      Triton FlashAttention structure over paged KV with `tl.dot`. Only a
#              candidate on devices where `tl.dot` lowers to mma (sm_80+); on the T4 it
#              compiles to FMA, spills, and runs one block per SM. Check with
#              `benchmarks/kernels/prefill_attention_ab.py --ptx-only` before enabling.
PREFILL_ATTENTION_KINDS = ("per_token", "sdpa", "tiled")


def _prefill_context_bucket(total_len: int, floor: int = 256) -> int:
    """Gathered-prefix length a captured SDPA prefill graph is shaped for."""
    bucket = floor
    while bucket < total_len:
        bucket *= 2
    return bucket
from engine.runtime import GenerationRequest, RequestState
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
    key_scale_pool: list | None = None
    value_scale_pool: list | None = None


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
    # One of PREFILL_ATTENTION_KINDS. See the note at the top of the module.
    attention: str = "sdpa"
    # Longest start + chunk in this batch, known host-side at planning time. The SDPA
    # path gathers this many logical tokens per row; reading it from the device tensors
    # would be a synchronization per layer.
    total_len: int = 0
    prefill_block_m: int | None = None
    prefill_block_n: int | None = None
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
        return paged_decode_batched(
            query, key_pool, value_pool, ctx.block_tables, ctx.seq_lens,
            scale=scaling, block_n=ctx.decode_block_n, num_warps=ctx.decode_num_warps,
            length_offset=1,
        )
    from engine.kernels.int8_paged_kv import paged_decode_batched_int8, write_decode_int8_kv
    write_decode_int8_kv(
        key, value, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
        ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.seq_lens,
    )
    return paged_decode_batched_int8(
        query, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
        ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.seq_lens,
        scale=scaling, block_n=ctx.decode_block_n, num_warps=ctx.decode_num_warps,
        length_offset=1,
    )


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
        if ctx.attention == "sdpa":
            out = sdpa_paged_prefill(
                query, key_pool, value_pool, ctx.block_tables,
                ctx.start_positions, ctx.chunk_lens, scale=scaling,
                total_len=ctx.total_len, cache=ctx.sdpa_cache,
            )
        elif ctx.attention == "tiled":
            out = tiled_paged_prefill(
                query, key_pool, value_pool, ctx.block_tables,
                ctx.start_positions, ctx.chunk_lens, scale=scaling,
                block_m=ctx.prefill_block_m, block_n=ctx.prefill_block_n,
            )
        else:
            out = paged_prefill(
                query, key_pool, value_pool, ctx.block_tables,
                ctx.start_positions, ctx.chunk_lens, scale=scaling,
            )
    else:
        from engine.kernels.int8_paged_kv import paged_prefill_int8, write_prefill_int8_kv_batched
        write_prefill_int8_kv_batched(
            key, value, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
            ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.chunk_lens,
            ctx.start_positions,
        )
        out = paged_prefill_int8(
            query, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
            ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.start_positions,
            ctx.chunk_lens, scale=scaling,
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
    assert total == decode_rows + prefill_rows * width, "fused row layout mismatch"
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
            # [1, h, B*W, D] -> [B, h, W, D]
            h = tensor.shape[1]
            return tensor[0, :, decode_rows:].reshape(h, prefill_rows, width, head_dim) \
                .transpose(0, 1).contiguous()

        out = _prefill_attention(ctx, layer_idx, rows(query), rows(key), rows(value), scaling)
        parts.append(out.transpose(1, 2).reshape(1, prefill_rows * width, heads, head_dim))
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    return out.contiguous(), None


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

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
                 tiled_prefill: bool = False,
                 prefill_block_m: int | None = None,
                 prefill_block_n: int | None = None,
                 max_prefill_tokens_per_iteration: int = 128,
                 max_waiting_requests: int | None = None,
                 prefix_cache_blocks: int = 256,
                 kv_cache_dtype: str = "fp16",
                 cuda_graph_batch_size: int | None = None,
                 cuda_graph_batch_sizes: tuple[int, ...] | None = None,
                 prefill_cuda_graphs: bool = True,
                 fused_step: bool = True,
                 fuse_mlp_gate_up: bool = False,
                 triton_rmsnorm: bool = True,
                 triton_rope: bool = True,
                 triton_swiglu: bool = True):
        if min(num_blocks, block_size, max_active, prefill_chunk_size,
               max_prefill_tokens_per_iteration) <= 0:
            raise ValueError("engine sizes and prefill budgets must be positive")
        if prefix_cache_blocks < 0:
            raise ValueError("prefix_cache_blocks must be non-negative")
        if kv_cache_dtype not in {"fp16", "int8"}:
            raise ValueError("kv_cache_dtype must be 'fp16' or 'int8'")
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
        # `tiled_prefill` is the older boolean form of the same choice. Default: SDPA over
        # gathered pages - measured on the T4 (journal, "SDPA under graphs") at -40% chat
        # and -66% long prefill step against the per-token kernel, and the only chunked
        # path that matches stock Transformers token-for-token on the identity prompts.
        if prefill_attention is None:
            prefill_attention = "tiled" if tiled_prefill else "sdpa"
        if prefill_attention not in PREFILL_ATTENTION_KINDS:
            raise ValueError(f"prefill_attention must be one of {PREFILL_ATTENTION_KINDS}")
        self.prefill_attention = prefill_attention
        self.tiled_prefill = prefill_attention == "tiled"
        self.prefill_block_m = prefill_block_m
        self.prefill_block_n = prefill_block_n
        self.max_prefill_tokens_per_iteration = max_prefill_tokens_per_iteration
        self.max_waiting_requests = max_waiting_requests
        self.prefix_cache_blocks = prefix_cache_blocks
        self.kv_cache_dtype = kv_cache_dtype
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or ()
        # Chunked prefill forwards are captured per (row bucket, kind, context bucket)
        # when graph buckets are configured. Off keeps decode graphs and runs prefill
        # eagerly on the same staged buffers, which is the A/B for the capture itself.
        self.prefill_cuda_graphs = prefill_cuda_graphs
        # A step that carries both decode rows and prefill chunks runs them as one packed
        # forward (decode tokens first, chunk batch after) instead of two. The weights
        # are read once for both, and the prefill's launch cost rides on the decode
        # forward's. Off runs the two forwards back to back, which is the A/B.
        self.fused_step = fused_step
        self.fuse_mlp_gate_up = fuse_mlp_gate_up
        self.triton_rmsnorm = triton_rmsnorm
        self.triton_rope = triton_rope
        self.triton_swiglu = triton_swiglu
        self._decode_graphs = {}

        cfg = model.config
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
        from engine.kernels.rope import install_triton_qwen_rope, uninstall_triton_qwen_rope
        from engine.kernels.swiglu import install_triton_qwen_swiglu, uninstall_triton_qwen_swiglu
        if triton_rmsnorm:
            self.triton_rmsnorm_modules = install_triton_rmsnorm(model)
        else:
            uninstall_triton_rmsnorm(model)
            self.triton_rmsnorm_modules = 0
        if triton_rope:
            install_triton_qwen_rope()
        else:
            uninstall_triton_qwen_rope()
        if fuse_mlp_gate_up and not triton_swiglu:
            raise ValueError("fuse_mlp_gate_up requires triton_swiglu=True")
        # Re-install when the fusion mode changes: the installer is a no-op on an
        # already patched module, so a prior engine's choice would otherwise persist.
        uninstall_triton_qwen_swiglu(model)
        if triton_swiglu:
            self.triton_swiglu_modules = install_triton_qwen_swiglu(
                model, fuse_gate_up=fuse_mlp_gate_up,
            )
        else:
            self.triton_swiglu_modules = 0

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
        else:
            self.key_scale_pool = None
            self.value_scale_pool = None

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
        required = max(self.cuda_graph_batch_sizes) - 1
        if required <= 0:
            return
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
                self.scheduler.preempt(request.request_id)
                return False
            self.scheduler.preempt(victim.request_id)

    def _capacity_or_fail(self, request: GenerationRequest, target_length: int) -> bool:
        """Acquire capacity; fail the request only if preemption could not help it."""
        if self._acquire_capacity(request, target_length):
            return True
        if request.state is not RequestState.WAITING:
            self.scheduler.fail(request.request_id, "KV_POOL_EXHAUSTED")
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
        }

    def _publish_prefix(
        self, request: GenerationRequest, next_token_id: int | None,
        token_ids: list[int] | None = None,
    ) -> None:
        sequence = request.prompt_token_ids if token_ids is None else token_ids
        if request.allocation is not None and sequence:
            self.prefix_cache.publish(sequence, request.allocation, next_token_id)

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
            self.scheduler.fail(request.request_id, "INVALID_PREFIX_ENTRY")
            return
        request.next_token_id = int(predicted_token)
        self._publish_prefix(request, request.next_token_id)
        self.scheduler.mark_decoding(request.request_id)
        request.append_token(request.next_token_id)
        if request.next_token_id in self.eos_ids or len(request.output_token_ids) >= request.max_new_tokens:
            reason = "EOS" if request.next_token_id in self.eos_ids else "LENGTH"
            self.scheduler.finish(request.request_id, reason=reason)

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
            self.key_scale_pool, self.value_scale_pool,
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
        next_tokens = self._lm_head()(hidden[rows, last_positions]).argmax(dim=-1).tolist()
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
            context_len = _prefill_context_bucket(total_len) if self.prefill_attention == "sdpa" else 0
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
                next_tokens = graph.replay()
            else:
                next_tokens = self._prefill_forward(row_count, width)
            # One device-to-host transfer for the batch.
            tokens = next_tokens[:count].tolist()
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

    def _set_prefill_context(self, row_count: int, total_len: int) -> None:
        _set_prefill_ctx(_PrefillContext(
            self.key_pool, self.value_pool,
            self._prefill_device_block_tables[:row_count],
            self._prefill_device_starts[:row_count],
            self._prefill_device_chunk_lens[:row_count],
            self.key_scale_pool, self.value_scale_pool,
            attention=self.prefill_attention,
            total_len=total_len,
            prefill_block_m=self.prefill_block_m,
            prefill_block_n=self.prefill_block_n,
        ))

    def _prefill_forward(self, row_count: int, width: int) -> torch.Tensor:
        """One chunked prefill forward over the staged buffers -> next token per row.

        The vocabulary projection runs on each row's last chunk token only; a row that
        does not finish its prompt this step simply discards the result.
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
        return self._lm_head()(hidden[rows, last]).argmax(dim=-1)

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
        return self.scheduler.plan_prefill(
            chunk_size=self.prefill_chunk_size,
            token_budget=self.max_prefill_tokens_per_iteration,
        )

    def cancel(self, request_id: str, reason: str = "CANCELLED_BY_CLIENT") -> GenerationRequest:
        """Cancel queued, partially-prefilled, or decoding work and release its KV."""
        return self.scheduler.cancel(request_id, reason=reason)

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

    def _commit_decode(self, active: list[GenerationRequest], tokens: list[int]) -> None:
        for s, token in zip(active, tokens):
            self.block_manager.append_tokens(s.request_id)
            tok = int(token)
            s.next_token_id = tok
            s.append_token(tok)
            if tok in self.eos_ids or len(s.output_token_ids) >= s.max_new_tokens:
                reason = "EOS" if tok in self.eos_ids else "LENGTH"
                self.scheduler.finish(s.request_id, reason=reason)

    def _set_fused_contexts(
        self, *, decode_rows: int, block_tables, seq_lens, block_n: int, num_warps: int,
        prefill_rows: int, total_len: int, width: int,
    ) -> None:
        _set_batch_ctx(_BatchContext(
            key_pool=self.key_pool, value_pool=self.value_pool,
            block_tables=block_tables, seq_lens=seq_lens, block_size=self.block_size,
            decode_block_n=block_n, decode_num_warps=num_warps,
            key_scale_pool=self.key_scale_pool, value_scale_pool=self.value_scale_pool,
        ))
        self._set_prefill_context(prefill_rows, total_len)
        _set_fused_ctx(_FusedContext(decode_rows, prefill_rows, width))

    @staticmethod
    def _clear_fused_contexts() -> None:
        _clear_fused_ctx()
        _clear_prefill_ctx()
        _clear_batch_ctx()

    def _fused_forward(self, decode_rows: int, prefill_rows: int, width: int) -> torch.Tensor:
        """One packed forward over the staged decode and prefill buffers -> next tokens.

        Returns `[decode_rows + prefill_rows]` greedy tokens: the decode rows' next
        tokens, then each chunk row's prediction at its last valid token (discarded by
        the caller for rows whose prompt is not finished). The vocabulary projection runs
        on exactly those rows.
        """
        decode_ids = self._device_input_ids[:decode_rows].reshape(1, decode_rows)
        decode_positions = self._device_position_ids[:decode_rows].reshape(1, decode_rows)
        chunk_ids = self._prefill_device_input_ids[:prefill_rows, :width].reshape(1, -1)
        chunk_positions = self._prefill_device_position_ids[:prefill_rows, :width].reshape(1, -1)
        input_ids = torch.cat((decode_ids, chunk_ids), dim=1)
        position_ids = torch.cat((decode_positions, chunk_positions), dim=1)
        hidden = self._decoder()(
            input_ids=input_ids, position_ids=position_ids, use_cache=False, return_dict=True,
        ).last_hidden_state[0]
        chunk_lens = self._prefill_device_chunk_lens[:prefill_rows]
        last = (
            decode_rows
            + torch.arange(prefill_rows, device=hidden.device) * width
            + (chunk_lens.to(torch.long) - 1).clamp_(min=0)
        )
        picked = torch.cat((hidden[:decode_rows], hidden[last]), dim=0)
        return self._lm_head()(picked).argmax(dim=-1)

    def _capture_fused_graph(self, key: tuple):
        """Capture one fused step graph, or record that this attention kind cannot be."""
        from engine.graphs.fused_step_graph import capture_fused_step_graph

        decode_rows, prefill_rows, kind, context_len, block_n, num_warps = key
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
        block_n, num_warps = select_paged_decode_config(max_sequence_length, decode_count)
        decode_bucket = self._graph_bucket(decode_count)
        prefill_bucket = self._prefill_row_bucket(prefill_count, self.FUSED_PREFILL_ROW_LIMIT)
        total_len = max(request.prefilled_token_count + n for request, n in plans)
        use_graph = (
            self.prefill_cuda_graphs and decode_bucket is not None and prefill_bucket is not None
            and f"fused:{self.prefill_attention}" not in self._prefill_graph_unsupported
        )
        graph = None
        if use_graph:
            width = self.prefill_chunk_size
            context_len = _prefill_context_bucket(total_len) if self.prefill_attention == "sdpa" else 0
            key = (decode_bucket, prefill_bucket, self.prefill_attention, context_len, block_n, num_warps)
            graph = self._fused_graphs.get(key)
            if graph is None:
                self.lazy_graph_captures += 1
                graph = self._capture_fused_graph(key)
            use_graph = graph is not None
        if use_graph:
            decode_rows, prefill_rows = decode_bucket, prefill_bucket
        else:
            decode_rows, prefill_rows = decode_count, prefill_count
            width = max(n for _, n in plans)
            context_len = total_len
        self._set_attention(self.FUSED_ATTN_NAME)
        _, _, block_tables, seq_lens = self._prepare_decode_metadata(
            rows, graph_bucket_size=decode_rows if use_graph else None,
        )
        self._prepare_prefill_metadata(plans, prefill_rows)
        self._host_elapsed(host_started, "host_stage_ms")
        timer = self._gpu_timer()
        self._set_fused_contexts(
            decode_rows=decode_rows, block_tables=block_tables, seq_lens=seq_lens,
            block_n=block_n, num_warps=num_warps, prefill_rows=prefill_rows,
            total_len=context_len if context_len else total_len, width=width,
        )
        try:
            if use_graph:
                next_tokens = graph.replay()
            else:
                next_tokens = self._fused_forward(decode_rows, prefill_rows, width)
            # One device-to-host transfer for decode and prefill together.
            sync_started = perf_counter() if self.instrument else None
            tokens = next_tokens.tolist()
            self._host_elapsed(sync_started, "sync_ms")
        finally:
            self._clear_fused_contexts()
        self._gpu_elapsed(timer, "fused_gpu_ms")
        self._commit_decode(rows, tokens[:decode_count])
        self._commit_prefill(plans, tokens[decode_rows:decode_rows + prefill_count])
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
            max_sequence_length, len(active)
        )
        graph_bucket_size = next(
            (
                size for size in self.cuda_graph_batch_sizes
                if len(active) <= size <= self.max_active
            ),
            None,
        )
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
            key_scale_pool=self.key_scale_pool, value_scale_pool=self.value_scale_pool,
        )
        graph_key = (graph_bucket_size, decode_block_n, decode_num_warps)
        use_graph = graph_bucket_size is not None
        if use_graph:
            graph = self._decode_graphs.get(graph_key)
            if graph is None:
                from engine.graphs import capture_paged_decode_graph
                self.lazy_graph_captures += 1
                graph = capture_paged_decode_graph(
                    self, batch_size=graph_bucket_size, block_n=decode_block_n,
                    num_warps=decode_num_warps,
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
        next_tokens = logits[:len(active), -1, :].argmax(dim=-1).tolist()
        self._host_elapsed(sync_started, "sync_ms")
        self._gpu_elapsed(timer, "decode_gpu_ms")
        self._commit_decode(active, next_tokens)

    # ------------------------------------------------------------------
    # D3: the continuous loop
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int = 32) -> list[list[int]]:
        """Run all prompts through continuous batching. Returns output token ids per prompt."""
        requests = []
        for i, p in enumerate(prompts):
            ids = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            request = GenerationRequest(
                request_id=f"seq{i}",
                prompt_token_count=len(ids),
                max_new_tokens=max_new_tokens,
                prompt_token_ids=ids,
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
                contexts = [_prefill_context_bucket(length) for length in (256, 512, 1024, 2048)]
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
            if self.fused_step:
                fused_rows = [
                    rows for rows in (1,) + self.cuda_graph_batch_sizes
                    if self._prefill_row_bucket(rows, self.FUSED_PREFILL_ROW_LIMIT) == rows
                ]
                for decode_rows in self.cuda_graph_batch_sizes:
                    for prefill_rows in fused_rows:
                        for context_len in contexts:
                            for block_n, num_warps in ((64, 4), (128, 4)):
                                key = (decode_rows, prefill_rows, self.prefill_attention,
                                       context_len, block_n, num_warps)
                                if key not in self._fused_graphs:
                                    if self._capture_fused_graph(key) is None:
                                        break
        summary = {
            "rounds": rounds,
            "graphs": len(self._decode_graphs),
            "prefill_graphs": len(self._prefill_graphs),
            "fused_graphs": len(self._fused_graphs),
            "prefill_graph_unsupported": dict(self._prefill_graph_unsupported),
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
