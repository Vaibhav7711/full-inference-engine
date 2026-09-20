"""M2: paged KV storage path (single request, batch=1).

Where M1 verified block addressing *in parallel* while HuggingFace's DynamicCache still
owned storage, M2 makes our block-structured pages the *authoritative* physical home of
the KV cache.

Interception point (verified for transformers 5.16.1):
    Qwen3Attention.forward -> past_key_values.update(k, v, layer_idx)
        -> DynamicCache.update -> self.layers[layer_idx].update(k, v)
    The *layer* is where K,V physically live (.keys / .values via torch.cat).
    We replace the layer's storage with a block-structured page tensor.

Design:
    PagedLayer   — owns one page tensor [num_blocks, block_size, num_kv_heads, head_dim]
                   per K and per V. update() writes new tokens into the next free slots
                   and returns a vectorized gather of all stored tokens.
    PagedCache   — subclasses DynamicCache, holds PagedLayers instead of DynamicLayers.
                   Attention calls it with the identical interface.

Honest scope of M2-minimal:
    - Single sequence, batch = 1. Multi-sequence shared-pool support is M2-full.
    - Our page tensors are the SOLE physical storage of K,V (not a parallel copy).
    - Gather is vectorized (index_select + reshape), no Python loop. This is what
      collapses M1's ~51% naive-gather overhead.
    - Blocks grow on demand at block boundaries, like real paged attention.

Tensor contract (Qwen3-0.6B): K,V per layer are [batch=1, num_kv_heads=8, seq, head_dim=128].
GQA repeat to query heads happens downstream in attention, unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache


def query_length_from(cache_position) -> int:
    """Number of new tokens in this forward pass, from either cache-interface generation.

    transformers >= 4.53 hands `get_mask_sizes` the `cache_position` tensor; older builds
    passed the query length as an int. The mask builder then evaluates
    `kv_length + kv_offset - width > 0`, which raises on a multi-element tensor, so the
    result must be a plain int either way.
    """
    if hasattr(cache_position, "shape"):
        return int(cache_position.shape[-1])
    return int(cache_position)


class PagedLayer:
    """Block-structured KV storage for one transformer layer, batch=1.

    Physical layout:
        key_pages   : [num_blocks, block_size, num_kv_heads, head_dim]
        value_pages : [num_blocks, block_size, num_kv_heads, head_dim]

    Logical token t lives at (block = t // block_size, offset = t % block_size).
    Blocks are allocated contiguously here (0, 1, 2, ...) for the single-sequence case;
    the non-contiguous block-table indirection is exercised in M1 and will return in
    M2-full's shared pool. What M2-minimal proves is that OUR pages, not HF's tensor,
    are the physical store — and that the vectorized write/gather is correct and fast.
    """

    def __init__(self, block_size_tokens: int = 16, initial_blocks: int = 4):
        self.block_size_tokens = block_size_tokens
        self.seq_len = 0                       # logical tokens stored so far
        self.key_pages: Optional[torch.Tensor] = None
        self.value_pages: Optional[torch.Tensor] = None
        self._initial_blocks = initial_blocks
        # Introspection counters (used by tests / benchmarks)
        self.num_growths = 0

    # -- lazy allocation on first write, once we know shape/dtype/device --
    def _init_pages(self, ref: torch.Tensor) -> None:
        batch, num_kv_heads, _, head_dim = ref.shape
        assert batch == 1, "M2-minimal supports batch=1 only"
        self.key_pages = ref.new_zeros(
            (self._initial_blocks, self.block_size_tokens, num_kv_heads, head_dim)
        )
        self.value_pages = torch.zeros_like(self.key_pages)

    @property
    def num_blocks(self) -> int:
        return 0 if self.key_pages is None else self.key_pages.shape[0]

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size_tokens

    def _grow_to(self, required_tokens: int) -> None:
        """Add physical blocks until capacity covers required_tokens."""
        while self.capacity_tokens < required_tokens:
            # Double the block count (amortized O(1) growth), like a dynamic array.
            add = max(self._initial_blocks, self.num_blocks)
            extra_k = self.key_pages.new_zeros(
                (add, *self.key_pages.shape[1:])
            )
            extra_v = torch.zeros_like(extra_k)
            self.key_pages = torch.cat([self.key_pages, extra_k], dim=0)
            self.value_pages = torch.cat([self.value_pages, extra_v], dim=0)
            self.num_growths += 1

    def _write(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Scatter new tokens [1, H, N, D] into pages at positions [seq_len, seq_len+N)."""
        num_new = key_states.shape[2]
        start = self.seq_len
        end = start + num_new
        if end > self.capacity_tokens:
            self._grow_to(end)

        # Transpose incoming [1, H, N, D] -> [N, H, D] (drop batch=1)
        k = key_states[0].transpose(0, 1)      # [N, H, D]
        v = value_states[0].transpose(0, 1)    # [N, H, D]

        # Compute (block, offset) for each new logical position and scatter.
        # For contiguous single-sequence layout this is a simple reshape when aligned,
        # but we handle the general (possibly block-straddling) case explicitly.
        for i in range(num_new):
            tok = start + i
            blk, off = divmod(tok, self.block_size_tokens)
            self.key_pages[blk, off] = k[i]
            self.value_pages[blk, off] = v[i]

        self.seq_len = end

    def _gather(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized gather of all stored tokens -> contiguous [1, H, seq_len, D].

        No Python loop over tokens. Flatten pages to [num_blocks*block_size, H, D],
        slice to seq_len, then reshape to the [batch=1, H, seq_len, D] attention wants.
        """
        # [num_blocks, block_size, H, D] -> [num_blocks*block_size, H, D]
        flat_k = self.key_pages.flatten(0, 1)[: self.seq_len]     # [S, H, D]
        flat_v = self.value_pages.flatten(0, 1)[: self.seq_len]
        # -> [1, H, S, D]
        keys = flat_k.transpose(0, 1).unsqueeze(0)
        values = flat_v.transpose(0, 1).unsqueeze(0)
        return keys, values

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DynamicLayer-compatible update: store new K,V in pages, return full K,V."""
        if self.key_pages is None:
            self._init_pages(key_states)
        self._write(key_states, value_states)
        return self._gather()

    # -- minimal DynamicLayer API surface the model / cache may touch --
    def get_seq_length(self, *args, **kwargs) -> int:
        return self.seq_len

    @property
    def keys(self) -> Optional[torch.Tensor]:
        if self.key_pages is None:
            return None
        return self._gather()[0]

    @property
    def values(self) -> Optional[torch.Tensor]:
        if self.value_pages is None:
            return None
        return self._gather()[1]


class PagedCache(DynamicCache):
    """DynamicCache whose per-layer storage is block-structured PagedLayers.

    Attention interacts with this exactly as it does with DynamicCache:
        keys, values = cache.update(key_states, value_states, layer_idx)
    but K,V now physically live in our page tensors, not HF's concatenated cache.
    """

    def __init__(self, num_layers: int, block_size_tokens: int = 16, initial_blocks: int = 4):
        # We deliberately do NOT call super().__init__ with layer replication;
        # we manage our own PagedLayer list.
        self.block_size_tokens = block_size_tokens
        self._paged_layers = [
            PagedLayer(block_size_tokens=block_size_tokens, initial_blocks=initial_blocks)
            for _ in range(num_layers)
        ]
        # Some DynamicCache internals inspect these; set safe defaults.
        self.layer_class_to_replicate = None
        try:
            super().__init__()
        except Exception:
            # If DynamicCache.__init__ requires args in this version, we bypass it;
            # our overrides below cover the methods the model actually calls.
            pass

    # -- route the two methods Qwen3 attention uses to our paged layers --

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._paged_layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0, *args, **kwargs) -> int:
        return self._paged_layers[layer_idx].get_seq_length()

    # -- mask/length methods the model's masking machinery calls --

    def get_mask_sizes(self, cache_position, layer_idx: int = 0) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for causal-mask construction.

        `cache_position` is the position tensor on transformers >= 4.53 and the query
        length as an int on older builds; both must yield plain ints here.

        The parent DynamicCache inspects layer types (CacheLayerMixin) which our
        PagedLayer is not, so its implementation raises. We answer directly from our
        stored sequence length.

        IMPORTANT: this is called BEFORE update() writes the new tokens for this pass.
        At call time, seq_len holds only previously-cached tokens, and query_length is
        the number of new tokens in this forward pass. The mask needs the total KV
        length that attention will see = past + new.
            kv_length = past_seen + query_length
            kv_offset = past_seen already attended before this pass (0 at prefill)
        """
        past_seen = self._paged_layers[layer_idx].seq_len
        kv_length = past_seen + query_length_from(cache_position)
        kv_offset = 0
        return kv_length, kv_offset

    def get_max_cache_shape(self, *args, **kwargs):
        """Unbounded (dynamic growth) cache -> no fixed max length."""
        return None

    def get_max_length(self, *args, **kwargs):
        return None

    def __len__(self) -> int:
        return len(self._paged_layers)

    # -- introspection --
    @property
    def layers(self):
        return self._paged_layers

    def total_blocks(self) -> int:
        return sum(l.num_blocks for l in self._paged_layers)

    def total_growths(self) -> int:
        return sum(l.num_growths for l in self._paged_layers)

    def snapshot(self) -> dict:
        l0 = self._paged_layers[0]
        return {
            "num_layers": len(self._paged_layers),
            "block_size_tokens": self.block_size_tokens,
            "seq_len": l0.seq_len,
            "blocks_per_layer": l0.num_blocks,
            "capacity_tokens_per_layer": l0.capacity_tokens,
            "total_blocks_all_layers": self.total_blocks(),
            "total_growths": self.total_growths(),
        }
