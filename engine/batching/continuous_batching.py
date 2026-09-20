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
from engine.kernels.tiled_paged_prefill import tiled_paged_prefill
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
    # Which fp16 prefill attention kernel to run. The tiled kernel loads each KV tile once
    # per tile of queries; the original loads the whole prefix once per query token. Kept
    # switchable so the two can be A/B'd in-engine and the old path stays available until
    # the new one has a token-identity gate behind it.
    tiled_prefill: bool = True
    prefill_block_m: int | None = None
    prefill_block_n: int | None = None


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
        if ctx.tiled_prefill:
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
    return out.transpose(1, 2).contiguous(), None


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """Full batched decode over paged KV, scheduler-driven."""

    ATTN_NAME = "batched_paged_decode"
    PREFILL_ATTN_NAME = "chunked_paged_prefill"
    # Gate 1B removed the preemption-count limit. A count is the wrong criterion: a
    # healthy request waiting behind several long generations yields once per iteration
    # through no fault of its own, so any fixed bound fails valid work under load.
    # Termination is now structural - see FCFSScheduler's policy docstring.

    def __init__(self, model, tokenizer, device, *,
                 num_blocks: int = 4096, block_size: int = 16, max_active: int = 16,
                 prefill_chunk_size: int = 128,
                 tiled_prefill: bool = True,
                 prefill_block_m: int | None = None,
                 prefill_block_n: int | None = None,
                 max_prefill_tokens_per_iteration: int = 128,
                 max_waiting_requests: int | None = None,
                 prefix_cache_blocks: int = 256,
                 kv_cache_dtype: str = "fp16",
                 cuda_graph_batch_size: int | None = None,
                 cuda_graph_batch_sizes: tuple[int, ...] | None = None,
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
        self.tiled_prefill = tiled_prefill
        self.prefill_block_m = prefill_block_m
        self.prefill_block_n = prefill_block_n
        self.max_prefill_tokens_per_iteration = max_prefill_tokens_per_iteration
        self.max_waiting_requests = max_waiting_requests
        self.prefix_cache_blocks = prefix_cache_blocks
        self.kv_cache_dtype = kv_cache_dtype
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or ()
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
            if len(self._graph_dummy_blocks) < row_count - count:
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
        for row in range(count, row_count):
            self._host_block_tables[row, 0] = self._graph_dummy_blocks[row - count]

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
        out = self.model(
            input_ids=ids, attention_mask=attention_mask,
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        rows = torch.arange(len(requests), device=self.device)
        last_positions = seq_lens.to(dtype=torch.long) - 1
        next_tokens = out.logits[rows, last_positions].argmax(dim=-1).tolist()
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

        viable_plans = []
        for request, count in plans:
            if request.state is not RequestState.PREFILLING:
                continue  # preempted while making room for an earlier request
            target = request.prefilled_token_count + count
            if self._capacity_or_fail(request, target):
                viable_plans.append((request, count))
        plans = [
            (request, count) for request, count in viable_plans
            if request.state is RequestState.PREFILLING and request.allocation is not None
        ]
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
            chunk = request.prefill_token_ids[start:start + count]
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
            tiled_prefill=self.tiled_prefill,
            prefill_block_m=self.prefill_block_m,
            prefill_block_n=self.prefill_block_n,
        ))
        timer = self._gpu_timer()
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
                raise RuntimeError("prefill capacity was acquired but could not be committed")
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
                self._complete_prefill(request, int(token))
        self._gpu_elapsed(timer, "prefill_gpu_ms")

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
        if decoding:
            self.decode_step(decoding)
        admitted = self.scheduler.admit_available(max_active_requests=self.max_active)
        for request in admitted:
            if request.remaining_prefill_tokens == 0:
                self._complete_prefill(request, request.cached_next_token_id)
        plans = self._plan_prefill_chunks()
        self.last_step_prefill_tokens = sum(count for _, count in plans)
        self.last_step_decode_rows = len(decoding)
        self.last_step_prefill_path = ""
        if plans:
            self.prefill_chunks(plans)
            self.prefill_steps += 1
        elif decoding:
            self.decode_only_steps += 1

    # ------------------------------------------------------------------
    # D2: one batched decode step over all active sequences
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def decode_step(self, active: list[GenerationRequest]) -> None:
        """Advance all active sequences by one token via a single batched forward."""
        N = len(active)
        if N == 0:
            return
        host_started = perf_counter() if self.instrument else None

        # Switch to the batched K4 attention fn
        self.model.config._attn_implementation = self.ATTN_NAME
        if hasattr(self.model.config, "_attn_implementation_internal"):
            self.model.config._attn_implementation_internal = self.ATTN_NAME

        # Grow in FCFS priority order. A request that cannot grow preempts newer requests
        # (which then vanish from this batch) and is failed only when nothing older can
        # help. Victims are always later in priority order, but a resumed request sits at
        # the end of the admission order with an old arrival time, so re-check at the end.
        viable = []
        for s in sorted(active, key=lambda item: (item.created_ns, item.request_id)):
            if s.state is not RequestState.DECODING or s.allocation is None:
                continue  # preempted earlier in this loop
            if self._capacity_or_fail(s, s.allocation.sequence_length + 1):
                viable.append(s)
        active = [s for s in viable if s.state is RequestState.DECODING and s.allocation is not None]
        if not active:
            return

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
        for s, token in zip(active, next_tokens):
            self.block_manager.append_tokens(s.request_id)
            tok = int(token)
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
        summary = {
            "rounds": rounds,
            "graphs": len(self._decode_graphs),
            "prefill_sdpa_calls": self.prefill_sdpa_calls,
            "prefill_chunked_calls": self.prefill_chunked_calls,
        }
        self.reset()
        self.prefill_steps = self.decode_only_steps = 0
        self.last_step_prefill_tokens = self.last_step_decode_rows = 0
        self.prefill_sdpa_calls = self.prefill_chunked_calls = 0
        self.prefill_sdpa_tokens = self.prefill_chunked_tokens = 0
        self.last_step_prefill_path = ""
        return summary

    def attn_call_count(self) -> int:
        return _ATTN_CALLS
