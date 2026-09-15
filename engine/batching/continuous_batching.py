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
from typing import Optional

import torch

from engine.cache import KVBlockManager, PrefixCache
from engine.kernels.kv_write import write_decode_kv
from engine.kernels.paged_decode_batched import (
    paged_decode_batched,
)
from engine.kernels.paged_decode_config import select_paged_decode_config
from engine.kernels.paged_prefill import paged_prefill
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


_PREFILL_CTX: Optional[_PrefillContext] = None
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
    layer_idx = module.layer_idx

    N, num_q_heads, one, D = query.shape
    assert one == 1, "batched decode: 1 new token per sequence"

    key_pool = ctx.key_pool[layer_idx]      # [num_blocks, block_size, kv_heads, D]
    value_pool = ctx.value_pool[layer_idx]

    if ctx.key_scale_pool is None:
        write_decode_kv(key, value, key_pool, value_pool, ctx.block_tables, ctx.seq_lens)
        out = paged_decode_batched(
            query, key_pool, value_pool, ctx.block_tables, ctx.seq_lens,
            scale=scaling, block_n=ctx.decode_block_n, num_warps=ctx.decode_num_warps,
            length_offset=1,
        )
    else:
        from engine.kernels.int8_paged_kv import paged_decode_batched_int8, write_decode_int8_kv
        write_decode_int8_kv(
            key, value, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
            ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.seq_lens,
        )
        out = paged_decode_batched_int8(
            query, key_pool, value_pool, ctx.key_scale_pool[layer_idx],
            ctx.value_scale_pool[layer_idx], ctx.block_tables, ctx.seq_lens,
            scale=scaling, block_n=ctx.decode_block_n, num_warps=ctx.decode_num_warps,
            length_offset=1,
        )

    # HF expects [N, 1, heads, D] (transposed form)
    out = out.transpose(1, 2).contiguous()   # [N, 1, num_q_heads, D]
    return out, None


def chunked_prefill_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs,
):
    """Write a K/V chunk, then attend it to the paged prefix causally."""
    ctx = _PREFILL_CTX
    assert ctx is not None, "chunked prefill context not set"
    layer_idx = module.layer_idx
    key_pool = ctx.key_pool[layer_idx]
    value_pool = ctx.value_pool[layer_idx]
    if ctx.key_scale_pool is None:
        from engine.kernels.kv_write import write_prefill_kv_batched
        write_prefill_kv_batched(
            key, value, key_pool, value_pool, ctx.block_tables,
            ctx.chunk_lens, ctx.start_positions,
        )
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
    return out.transpose(1, 2).contiguous(), None


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """Full batched decode over paged KV, scheduler-driven."""

    ATTN_NAME = "batched_paged_decode"
    PREFILL_ATTN_NAME = "chunked_paged_prefill"

    def __init__(self, model, tokenizer, device, *,
                 num_blocks: int = 4096, block_size: int = 16, max_active: int = 16,
                 prefill_chunk_size: int = 128,
                 max_prefill_tokens_per_iteration: int = 512,
                 max_waiting_requests: int | None = None,
                 prefix_cache_blocks: int = 256,
                 kv_cache_dtype: str = "fp16"):
        if min(num_blocks, block_size, max_active, prefill_chunk_size,
               max_prefill_tokens_per_iteration) <= 0:
            raise ValueError("engine sizes and prefill budgets must be positive")
        if prefix_cache_blocks < 0:
            raise ValueError("prefix_cache_blocks must be non-negative")
        if kv_cache_dtype not in {"fp16", "int8"}:
            raise ValueError("kv_cache_dtype must be 'fp16' or 'int8'")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.max_active = max_active
        self.prefill_chunk_size = prefill_chunk_size
        self.max_prefill_tokens_per_iteration = max_prefill_tokens_per_iteration
        self.max_waiting_requests = max_waiting_requests
        self.prefix_cache_blocks = prefix_cache_blocks
        self.kv_cache_dtype = kv_cache_dtype

        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.num_q_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

        # Qwen uses RMSNorm for hidden states and for per-head Q/K normalization.
        # Install one FP32-accumulating Triton kernel for both shapes before warmup.
        from engine.kernels.rmsnorm import install_triton_rmsnorm
        from engine.kernels.rope import install_triton_qwen_rope
        from engine.kernels.swiglu import install_triton_qwen_swiglu
        self.triton_rmsnorm_modules = install_triton_rmsnorm(model)
        install_triton_qwen_rope()
        self.triton_swiglu_modules = install_triton_qwen_swiglu(model)

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
        self.block_manager = KVBlockManager(
            num_blocks=num_blocks, block_size_tokens=block_size
        )
        self.prefix_cache = PrefixCache(self.block_manager, prefix_cache_blocks)
        self.scheduler = FCFSScheduler(
            self.block_manager, max_waiting_requests=max_waiting_requests,
            prefix_cache=self.prefix_cache,
        )

        # Register the batched attention fn once.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS[self.ATTN_NAME] = batched_decode_attention_forward
        ALL_ATTENTION_FUNCTIONS[self.PREFILL_ATTN_NAME] = chunked_prefill_attention_forward

    def _prepare_decode_metadata(
        self, active: list[GenerationRequest]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage one compact decode batch into persistent GPU metadata buffers."""
        count = len(active)
        if count > self.max_active:
            raise ValueError("active batch exceeds max_active")
        for row, request in enumerate(active):
            allocation = request.allocation
            if allocation is None or request.next_token_id is None:
                raise RuntimeError("decode request is missing token or KV allocation")
            self._host_input_ids[row, 0] = request.next_token_id
            self._host_position_ids[row, 0] = allocation.sequence_length
            self._host_seq_lens[row] = allocation.sequence_length
            for column, physical_block in enumerate(request.block_table):
                self._host_block_tables[row, column] = physical_block

        input_ids = self._device_input_ids[:count]
        position_ids = self._device_position_ids[:count]
        seq_lens = self._device_seq_lens[:count]
        # Keep the complete row width so this view is contiguous. The Triton kernel
        # indexes only blocks covered by seq_lens; unused columns are never read.
        block_tables = self._device_block_tables[:count]
        input_ids.copy_(self._host_input_ids[:count], non_blocking=True)
        position_ids.copy_(self._host_position_ids[:count], non_blocking=True)
        seq_lens.copy_(self._host_seq_lens[:count], non_blocking=True)
        block_tables.copy_(self._host_block_tables[:count], non_blocking=True)
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
        self.prefix_cache = PrefixCache(self.block_manager, self.prefix_cache_blocks)
        self.scheduler = FCFSScheduler(
            self.block_manager, max_waiting_requests=self.max_waiting_requests,
            prefix_cache=self.prefix_cache,
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
        for key_pool, value_pool in zip(self.key_pool, self.value_pool):
            key_pool[new_block].copy_(key_pool[old_block])
            value_pool[new_block].copy_(value_pool[old_block])
        if self.key_scale_pool is not None:
            for key_scale, value_scale in zip(self.key_scale_pool, self.value_scale_pool):
                key_scale[new_block].copy_(key_scale[old_block])
                value_scale[new_block].copy_(value_scale[old_block])
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

    def _publish_prefix(self, request: GenerationRequest, next_token_id: int) -> None:
        if request.allocation is not None and request.prompt_token_ids:
            self.prefix_cache.publish(
                request.prompt_token_ids, request.allocation, next_token_id
            )

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
            if self._ensure_kv_capacity(request, request.prompt_token_count):
                viable.append(request)
            else:
                self.scheduler.fail(request.request_id, "KV_POOL_EXHAUSTED")
        requests = viable
        if not requests:
            return

        lengths_list = [len(request.prompt_token_ids) for request in requests]
        padded_length = max(lengths_list)
        max_blocks = max(len(request.block_table) for request in requests)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = next(iter(self.eos_ids), 0)
        padded_ids = [
            request.prompt_token_ids + [pad_token_id] * (padded_length - length)
            for request, length in zip(requests, lengths_list)
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
        out = self.model(
            input_ids=ids, attention_mask=attention_mask,
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        rows = torch.arange(len(requests), device=self.device)
        last_positions = seq_lens.to(dtype=torch.long) - 1
        next_tokens = out.logits[rows, last_positions].argmax(dim=-1).tolist()

        for request, token in zip(requests, next_tokens):
            if not self.block_manager.append_tokens(
                request.request_id, request.remaining_prefill_tokens
            ):
                self.scheduler.fail(request.request_id, "KV_POOL_EXHAUSTED")
                continue
            request.advance_prefill(request.remaining_prefill_tokens)
            request.next_token_id = int(token)
            self._publish_prefix(request, request.next_token_id)
            self.scheduler.mark_decoding(request.request_id)
            request.append_token(request.next_token_id)
            if request.next_token_id in self.eos_ids or len(request.output_token_ids) >= request.max_new_tokens:
                reason = "EOS" if request.next_token_id in self.eos_ids else "LENGTH"
                self.scheduler.finish(request.request_id, reason=reason)

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
            request.prefilled_token_count == 0 and count == request.prompt_token_count
            for request, count in plans
        ):
            self.prefill_batch([request for request, _ in plans])
            return

        viable_plans = []
        for request, count in plans:
            target = request.prefilled_token_count + count
            if self._ensure_kv_capacity(request, target):
                viable_plans.append((request, count))
            else:
                self.scheduler.fail(request.request_id, "KV_POOL_EXHAUSTED")
        plans = viable_plans
        if not plans:
            return

        self.model.config._attn_implementation = self.PREFILL_ATTN_NAME
        if hasattr(self.model.config, "_attn_implementation_internal"):
            self.model.config._attn_implementation_internal = self.PREFILL_ATTN_NAME

        starts_list = [request.prefilled_token_count for request, _ in plans]
        counts_list = [count for _, count in plans]
        padded_length = max(counts_list)
        max_blocks = max(len(request.block_table) for request, _ in plans)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = next(iter(self.eos_ids), 0)
        padded_ids = []
        position_rows = []
        padded_tables = []
        for (request, count), start in zip(plans, starts_list):
            chunk = request.prompt_token_ids[start:start + count]
            padded_ids.append(chunk + [pad_token_id] * (padded_length - count))
            # Padded positions are not semantically observed, but valid absolute
            # positions are essential so RoPE agrees with future decode steps.
            positions = list(range(start, start + count))
            positions.extend([start] * (padded_length - count))
            position_rows.append(positions)
            padded_tables.append(
                request.block_table + [-1] * (max_blocks - len(request.block_table))
            )

        ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(position_rows, dtype=torch.long, device=self.device)
        block_tables = torch.tensor(padded_tables, dtype=torch.int32, device=self.device)
        starts = torch.tensor(starts_list, dtype=torch.int32, device=self.device)
        chunk_lens = torch.tensor(counts_list, dtype=torch.int32, device=self.device)
        _set_prefill_ctx(_PrefillContext(
            self.key_pool, self.value_pool, block_tables, starts, chunk_lens,
            self.key_scale_pool, self.value_scale_pool,
        ))
        try:
            out = self.model(
                input_ids=ids, position_ids=position_ids,
                use_cache=False, return_dict=True,
            )
        finally:
            _clear_prefill_ctx()

        completed_rows: list[tuple[int, GenerationRequest]] = []
        for row, (request, count) in enumerate(plans):
            if not self.block_manager.append_tokens(request.request_id, count):
                self.scheduler.fail(request.request_id, "KV_POOL_EXHAUSTED")
                continue
            request.advance_prefill(count)
            if request.remaining_prefill_tokens == 0:
                completed_rows.append((row, request))

        # Only the final prompt token produces the first generated token. Keeping this
        # device-side until one list transfer avoids a scalar synchronization per row.
        if completed_rows:
            rows = torch.tensor([row for row, _ in completed_rows], device=self.device)
            positions = torch.tensor(
                [counts_list[row] - 1 for row, _ in completed_rows], device=self.device
            )
            tokens = out.logits[rows, positions].argmax(dim=-1).tolist()
            for (_, request), token in zip(completed_rows, tokens):
                request.next_token_id = int(token)
                self._publish_prefix(request, request.next_token_id)
                self.scheduler.mark_decoding(request.request_id)
                request.append_token(request.next_token_id)
                if (request.next_token_id in self.eos_ids
                        or len(request.output_token_ids) >= request.max_new_tokens):
                    reason = "EOS" if request.next_token_id in self.eos_ids else "LENGTH"
                    self.scheduler.finish(request.request_id, reason=reason)

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
        """Run one decode-first scheduling iteration under the prefill budget."""
        decoding = [
            request for request in self.scheduler.active.values()
            if request.state is RequestState.DECODING
        ]
        if decoding:
            self.decode_step(decoding)
        admitted = self.scheduler.admit_available(max_active_requests=self.max_active)
        for request in admitted:
            if request.remaining_prefill_tokens == 0:
                if request.cached_next_token_id is None:
                    self.scheduler.fail(request.request_id, "INVALID_PREFIX_ENTRY")
                    continue
                request.next_token_id = request.cached_next_token_id
                self.scheduler.mark_decoding(request.request_id)
                request.append_token(request.next_token_id)
                if (request.next_token_id in self.eos_ids
                        or len(request.output_token_ids) >= request.max_new_tokens):
                    reason = "EOS" if request.next_token_id in self.eos_ids else "LENGTH"
                    self.scheduler.finish(request.request_id, reason=reason)
        plans = self._plan_prefill_chunks()
        if plans:
            self.prefill_chunks(plans)

    # ------------------------------------------------------------------
    # D2: one batched decode step over all active sequences
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def decode_step(self, active: list[GenerationRequest]) -> None:
        """Advance all active sequences by one token via a single batched forward."""
        N = len(active)
        if N == 0:
            return

        # Switch to the batched K4 attention fn
        self.model.config._attn_implementation = self.ATTN_NAME
        if hasattr(self.model.config, "_attn_implementation_internal"):
            self.model.config._attn_implementation_internal = self.ATTN_NAME

        # Fail only requests that cannot grow; unrelated sequences keep making progress.
        viable = []
        for s in active:
            target_length = s.allocation.sequence_length + 1
            if not self._ensure_kv_capacity(s, target_length):
                self.scheduler.fail(s.request_id, "KV_POOL_EXHAUSTED")
            else:
                viable.append(s)
        active = viable
        if not active:
            return

        input_ids, position_ids, block_tables, seq_lens = self._prepare_decode_metadata(active)
        max_sequence_length = max(
            request.allocation.sequence_length + 1 for request in active
        )
        decode_block_n, decode_num_warps = select_paged_decode_config(
            max_sequence_length, len(active)
        )

        # Stash context for the attention fn
        _set_batch_ctx(_BatchContext(
            key_pool=self.key_pool, value_pool=self.value_pool,
            block_tables=block_tables, seq_lens=seq_lens, block_size=self.block_size,
            decode_block_n=decode_block_n, decode_num_warps=decode_num_warps,
            key_scale_pool=self.key_scale_pool, value_scale_pool=self.value_scale_pool,
        ))
        try:
            out = self.model(input_ids=input_ids, position_ids=position_ids,
                             use_cache=False, return_dict=True)
        finally:
            _clear_batch_ctx()

        # Sample next token per sequence, advance state
        next_tokens = out.logits[:, -1, :].argmax(dim=-1)   # [N]
        for i, s in enumerate(active):
            self.block_manager.append_tokens(s.request_id)
            tok = int(next_tokens[i].item())
            s.next_token_id = tok
            s.append_token(tok)
            if tok in self.eos_ids or len(s.output_token_ids) >= s.max_new_tokens:
                reason = "EOS" if tok in self.eos_ids else "LENGTH"
                self.scheduler.finish(s.request_id, reason=reason)

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

    def attn_call_count(self) -> int:
        return _ATTN_CALLS
