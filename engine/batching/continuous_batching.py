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

from engine.cache.paging import KVBlockManager
from engine.kernels.kv_write import write_decode_kv
from engine.kernels.paged_decode_batched import paged_decode_batched
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


_BATCH_CTX: Optional[_BatchContext] = None
_ATTN_CALLS = 0


def _set_batch_ctx(ctx: _BatchContext) -> None:
    global _BATCH_CTX
    _BATCH_CTX = ctx


def _clear_batch_ctx() -> None:
    global _BATCH_CTX
    _BATCH_CTX = None


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

    # One kernel writes every sequence's new token. Keeping positions and block-table
    # lookups on-device avoids 2*N `.item()` synchronizations in every model layer.
    write_decode_kv(
        key, value, key_pool, value_pool, ctx.block_tables, ctx.seq_lens
    )

    # Run K4 over lengths including the token just written. The +1 is performed while
    # each attention program loads its length, rather than by a separate tensor kernel.
    out = paged_decode_batched(
        query,                 # [N, num_q_heads, 1, D]
        key_pool, value_pool,  # shared pool for this layer
        ctx.block_tables,      # [N, max_blocks]
        ctx.seq_lens,          # [N] lengths before the just-written token
        scale=scaling,
        block_n=64,
        length_offset=1,
    )   # -> [N, num_q_heads, 1, D]

    # HF expects [N, 1, heads, D] (transposed form)
    out = out.transpose(1, 2).contiguous()   # [N, 1, num_q_heads, D]
    return out, None


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """Full batched decode over paged KV, scheduler-driven."""

    ATTN_NAME = "batched_paged_decode"

    def __init__(self, model, tokenizer, device, *,
                 num_blocks: int = 4096, block_size: int = 16, max_active: int = 16):
        if num_blocks <= 0 or block_size <= 0 or max_active <= 0:
            raise ValueError("num_blocks, block_size, and max_active must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.max_active = max_active

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

        dtype = next(model.parameters()).dtype
        self.key_pool = [
            torch.zeros((num_blocks, block_size, self.num_kv_heads, self.head_dim),
                        device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self.value_pool = [torch.zeros_like(k) for k in self.key_pool]

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
        self.scheduler = FCFSScheduler(self.block_manager)

        # Register the batched attention fn once.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS[self.ATTN_NAME] = batched_decode_attention_forward

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
        self.scheduler = FCFSScheduler(self.block_manager)

    # ------------------------------------------------------------------
    # D1: prefill a sequence, store its (rotated) K,V into the pool
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, request: GenerationRequest) -> None:
        """Compatibility wrapper for a one-request batched prefill."""
        self.prefill_batch([request])

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
            self.key_pool, self.value_pool, block_tables, seq_lens, padded_length
        )
        out = self.model(
            input_ids=ids, attention_mask=attention_mask,
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        rows = torch.arange(len(requests), device=self.device)
        last_positions = seq_lens.to(dtype=torch.long) - 1
        next_tokens = out.logits[rows, last_positions].argmax(dim=-1).tolist()

        for request, token in zip(requests, next_tokens):
            request.next_token_id = int(token)
            self.scheduler.mark_decoding(request.request_id)
            request.append_token(request.next_token_id)
            if request.next_token_id in self.eos_ids or len(request.output_token_ids) >= request.max_new_tokens:
                reason = "EOS" if request.next_token_id in self.eos_ids else "LENGTH"
                self.scheduler.finish(request.request_id, reason=reason)

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

        # Ensure each sequence has a block for its new token (grow if at a boundary)
        for s in active:
            target_length = s.allocation.sequence_length + 1
            if not self.block_manager.ensure_capacity(s.request_id, target_length):
                raise RuntimeError(f"pool exhausted growing {s.request_id}")

        input_ids, position_ids, block_tables, seq_lens = self._prepare_decode_metadata(active)

        # Stash context for the attention fn
        _set_batch_ctx(_BatchContext(
            key_pool=self.key_pool, value_pool=self.value_pool,
            block_tables=block_tables, seq_lens=seq_lens, block_size=self.block_size,
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
            self.scheduler.submit(request)

        while self.scheduler.waiting or self.scheduler.active:
            admitted = self.scheduler.admit_available(max_active_requests=self.max_active)
            self.prefill_batch(admitted)
            active = [
                request
                for request in self.scheduler.active.values()
                if request.state is RequestState.DECODING
            ]
            if active:
                self.decode_step(active)

        return [request.output_token_ids for request in requests]

    def attn_call_count(self) -> int:
        return _ATTN_CALLS
