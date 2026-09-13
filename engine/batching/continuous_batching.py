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

from dataclasses import dataclass, field
from typing import Optional

import torch

from engine.cache.allocator import BlockAllocator
from engine.kernels.paged_decode_batched import paged_decode_batched


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
    kv_heads = key.shape[1]
    block_size = ctx.block_size

    key_pool = ctx.key_pool[layer_idx]      # [num_blocks, block_size, kv_heads, D]
    value_pool = ctx.value_pool[layer_idx]

    # --- Write each sequence's new K,V into the pool at its next slot ---
    # seq_lens[i] is the length BEFORE this token, so the new token goes at position
    # seq_lens[i] -> block = seq_lens[i] // block_size, offset = seq_lens[i] % block_size.
    for i in range(N):
        pos = int(ctx.seq_lens[i].item())
        lb = pos // block_size
        off = pos % block_size
        pblock = int(ctx.block_tables[i, lb].item())
        # key[i]: [kv_heads, 1, D] -> [kv_heads, D]
        key_pool[pblock, off] = key[i, :, 0, :]
        value_pool[pblock, off] = value[i, :, 0, :]

    # --- Run K4 batched decode: each query attends to its blocks [0 .. seq_lens[i]] ---
    # New per-sequence KV length INCLUDING this token = seq_lens + 1.
    kv_lens_now = ctx.seq_lens + 1
    out = paged_decode_batched(
        query,                 # [N, num_q_heads, 1, D]
        key_pool, value_pool,  # shared pool for this layer
        ctx.block_tables,      # [N, max_blocks]
        kv_lens_now,           # [N] lengths including the just-written token
        scale=scaling,
        block_n=64,
    )   # -> [N, num_q_heads, 1, D]

    # HF expects [N, 1, heads, D] (transposed form)
    out = out.transpose(1, 2).contiguous()   # [N, 1, num_q_heads, D]
    return out, None


# ---------------------------------------------------------------------------
# Sequence state
# ---------------------------------------------------------------------------

@dataclass
class SeqState:
    seq_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    block_table: list[int] = field(default_factory=list)
    seq_len: int = 0
    output_ids: list[int] = field(default_factory=list)
    next_token: Optional[int] = None
    done: bool = False


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """Full batched decode over paged KV, scheduler-driven."""

    ATTN_NAME = "batched_paged_decode"

    def __init__(self, model, tokenizer, device, *,
                 num_blocks: int = 4096, block_size: int = 16, max_active: int = 16):
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
        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size_tokens=block_size)

        # Register the batched attention fn once.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS[self.ATTN_NAME] = batched_decode_attention_forward

    def reset(self) -> None:
        """Reinitialize the allocator (fresh free-block list) for a clean run.

        Blocks are already released as sequences finish, but this guarantees a clean
        slate when reusing the same engine for multiple benchmark runs. The pool tensors
        are reused (not reallocated) — only the allocator's bookkeeping resets.
        """
        from engine.cache.allocator import BlockAllocator
        self.allocator = BlockAllocator(
            num_blocks=self.key_pool[0].shape[0], block_size_tokens=self.block_size,
        )

    # ------------------------------------------------------------------
    # D1: prefill a sequence, store its (rotated) K,V into the pool
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, seq: SeqState) -> None:
        """Prefill seq's prompt with stock attention, scatter K,V into pool, get 1st token."""
        from engine.cache.paged_cache import PagedCache

        # Use stock sdpa for prefill (correct, simple). We capture K,V from a PagedCache.
        self.model.config._attn_implementation = "sdpa"
        if hasattr(self.model.config, "_attn_implementation_internal"):
            self.model.config._attn_implementation_internal = "sdpa"

        ids = torch.tensor([seq.prompt_ids], device=self.device)
        prompt_len = ids.shape[1]

        nblocks = (prompt_len + self.block_size - 1) // self.block_size
        phys = self.allocator.allocate(seq.seq_id, nblocks)
        if phys is None:
            raise RuntimeError(f"pool exhausted prefilling {seq.seq_id}")
        seq.block_table = list(phys)

        scratch = PagedCache(num_layers=self.num_layers,
                             block_size_tokens=self.block_size, initial_blocks=nblocks + 1)
        out = self.model(input_ids=ids, past_key_values=scratch, use_cache=True, return_dict=True)

        # Scatter each layer's captured (rotated) K,V into the pool at seq's blocks
        for layer_idx in range(self.num_layers):
            pl = scratch._paged_layers[layer_idx]
            k_flat = pl.key_pages.flatten(0, 1)[:prompt_len]     # [prompt_len, kv_heads, D]
            v_flat = pl.value_pages.flatten(0, 1)[:prompt_len]
            for pos in range(prompt_len):
                lb = pos // self.block_size
                off = pos % self.block_size
                pb = seq.block_table[lb]
                self.key_pool[layer_idx][pb, off] = k_flat[pos]
                self.value_pool[layer_idx][pb, off] = v_flat[pos]

        seq.seq_len = prompt_len
        seq.next_token = int(out.logits[0, -1, :].argmax().item())
        seq.output_ids.append(seq.next_token)
        if seq.next_token in self.eos_ids or len(seq.output_ids) >= seq.max_new_tokens:
            seq.done = True

    # ------------------------------------------------------------------
    # D2: one batched decode step over all active sequences
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def decode_step(self, active: list[SeqState]) -> None:
        """Advance all active sequences by one token via a single batched forward."""
        N = len(active)
        if N == 0:
            return

        # Switch to the batched K4 attention fn
        self.model.config._attn_implementation = self.ATTN_NAME
        if hasattr(self.model.config, "_attn_implementation_internal"):
            self.model.config._attn_implementation_internal = self.ATTN_NAME

        # Build batched inputs
        input_ids = torch.tensor([[s.next_token] for s in active], device=self.device)  # [N,1]
        position_ids = torch.tensor([[s.seq_len] for s in active], device=self.device)  # [N,1]

        # Build block_tables [N, max_blocks] and seq_lens [N]
        max_blocks = max(len(s.block_table) for s in active)
        block_tables = torch.zeros((N, max_blocks), dtype=torch.int32, device=self.device)
        seq_lens = torch.zeros((N,), dtype=torch.int32, device=self.device)
        for i, s in enumerate(active):
            for j, pb in enumerate(s.block_table):
                block_tables[i, j] = pb
            seq_lens[i] = s.seq_len

        # Ensure each sequence has a block for its new token (grow if at a boundary)
        for i, s in enumerate(active):
            need_blocks = (s.seq_len + 1 + self.block_size - 1) // self.block_size
            if need_blocks > len(s.block_table):
                new = self.allocator.extend(s.seq_id, need_blocks - len(s.block_table))
                if new is None:
                    raise RuntimeError(f"pool exhausted growing {s.seq_id}")
                s.block_table.extend(new)
        # Rebuild block_tables after possible growth
        max_blocks = max(len(s.block_table) for s in active)
        block_tables = torch.zeros((N, max_blocks), dtype=torch.int32, device=self.device)
        for i, s in enumerate(active):
            for j, pb in enumerate(s.block_table):
                block_tables[i, j] = pb

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
            s.seq_len += 1
            tok = int(next_tokens[i].item())
            s.next_token = tok
            s.output_ids.append(tok)
            if tok in self.eos_ids or len(s.output_ids) >= s.max_new_tokens:
                s.done = True

    # ------------------------------------------------------------------
    # D3: the continuous loop
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int = 32) -> list[list[int]]:
        """Run all prompts through continuous batching. Returns output token ids per prompt."""
        # Build sequences
        seqs = []
        for i, p in enumerate(prompts):
            ids = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            seqs.append(SeqState(seq_id=f"seq{i}", prompt_ids=ids, max_new_tokens=max_new_tokens))

        waiting = list(seqs)
        active: list[SeqState] = []
        finished: list[SeqState] = []

        while waiting or active:
            # Admit and prefill new sequences up to max_active
            while waiting and len(active) < self.max_active:
                s = waiting.pop(0)
                self.prefill(s)
                if s.done:
                    self.allocator.release(s.seq_id)
                    finished.append(s)
                else:
                    active.append(s)

            if not active:
                continue

            # One batched decode step over all active
            self.decode_step(active)

            # Evict finished
            still_active = []
            for s in active:
                if s.done:
                    self.allocator.release(s.seq_id)
                    finished.append(s)
                else:
                    still_active.append(s)
            active = still_active

        # Return in original order
        by_id = {s.seq_id: s for s in finished}
        return [by_id[f"seq{i}"].output_ids for i in range(len(prompts))]

    def attn_call_count(self) -> int:
        return _ATTN_CALLS
