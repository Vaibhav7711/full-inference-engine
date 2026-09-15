"""Cache adapter that writes prefill K/V directly into the shared paged pool."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from engine.kernels.kv_write import write_prefill_kv_batched


class BatchedPoolBackedPrefillCache(DynamicCache):
    """Cache adapter for padded multi-request prefill into shared physical blocks."""

    def __init__(self, key_pool, value_pool, block_tables, seq_lens, padded_length):
        if not key_pool or len(key_pool) != len(value_pool):
            raise ValueError("matching non-empty per-layer K/V pools are required")
        self.key_pool = key_pool
        self.value_pool = value_pool
        self.block_tables = block_tables
        self.seq_lens = seq_lens
        self.padded_length = padded_length
        self._layer_lengths = [0] * len(key_pool)
        self.layer_class_to_replicate = None
        try:
            super().__init__()
        except Exception:
            pass

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self._layer_lengths[layer_idx] != 0:
            raise RuntimeError("batched prefill cache supports exactly one write per layer")
        write_prefill_kv_batched(
            key_states, value_states, self.key_pool[layer_idx], self.value_pool[layer_idx],
            self.block_tables, self.seq_lens,
        )
        self._layer_lengths[layer_idx] = self.padded_length
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0, *args, **kwargs) -> int:
        return self._layer_lengths[layer_idx]

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self._layer_lengths[layer_idx] + query_length, 0

    def get_max_cache_shape(self, *args, **kwargs):
        return None

    def get_max_length(self, *args, **kwargs):
        return None

    def __len__(self) -> int:
        return len(self._layer_lengths)
