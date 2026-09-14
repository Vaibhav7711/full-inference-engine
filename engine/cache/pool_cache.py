"""Cache adapter that writes prefill K/V directly into the shared paged pool."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from engine.kernels.kv_write import write_paged_kv


class PoolBackedPrefillCache(DynamicCache):
    """DynamicCache-compatible prefill writer without temporary KV ownership.

    Qwen attention passes already-rotated K/V to ``update``. This adapter writes them
    directly to the request's physical blocks and returns the original tensors for the
    current prefill attention operation. It is intentionally single-request/prefill
    only; continuous decode reads the shared pool through the K4 kernel.
    """

    def __init__(
        self,
        key_pool: list[torch.Tensor],
        value_pool: list[torch.Tensor],
        block_table: torch.Tensor,
    ):
        if not key_pool or len(key_pool) != len(value_pool):
            raise ValueError("matching non-empty per-layer K/V pools are required")
        self.key_pool = key_pool
        self.value_pool = value_pool
        self.block_table = block_table
        self._layer_lengths = [0] * len(key_pool)
        self.layer_class_to_replicate = None
        try:
            super().__init__()
        except Exception:
            # Transformers cache initialization has changed across supported releases;
            # all model-facing methods used by this adapter are overridden below.
            pass

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self._layer_lengths[layer_idx] != 0:
            raise RuntimeError("PoolBackedPrefillCache supports exactly one prefill write")
        write_paged_kv(
            key_states,
            value_states,
            self.key_pool[layer_idx],
            self.value_pool[layer_idx],
            self.block_table,
        )
        self._layer_lengths[layer_idx] = key_states.shape[2]
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0, *args, **kwargs) -> int:
        return self._layer_lengths[layer_idx]

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        past_seen = self._layer_lengths[layer_idx]
        return past_seen + query_length, 0

    def get_max_cache_shape(self, *args, **kwargs):
        return None

    def get_max_length(self, *args, **kwargs):
        return None

    def __len__(self) -> int:
        return len(self._layer_lengths)
