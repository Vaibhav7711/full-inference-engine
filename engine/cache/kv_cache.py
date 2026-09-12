"""Stage 4 KV-cache geometry and accounting.

This module models cache memory without owning the cache allocation.  In Stage 1/4,
Transformers still allocates the physical cache; later stages will replace that policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class KVCacheGeometry:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    num_query_heads: int | None = None

    @classmethod
    def from_model_config(cls, config: Any, dtype: torch.dtype) -> "KVCacheGeometry":
        num_layers = getattr(config, "num_hidden_layers", None)
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        num_attention_heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and getattr(config, "hidden_size", None) and num_attention_heads:
            head_dim = config.hidden_size // num_attention_heads
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads
        if not all(isinstance(value, int) and value > 0 for value in (num_layers, num_kv_heads, head_dim)):
            raise ValueError("model config must expose positive layer, KV-head, and head-dimension values")
        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            num_query_heads=num_attention_heads,
        )

    @property
    def dtype_bytes(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    @property
    def bytes_per_token(self) -> int:
        """One key and one value per layer, KV head, and head-dimension element."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def gqa_group_size(self) -> float | None:
        if self.num_query_heads is None:
            return None
        return self.num_query_heads / self.num_kv_heads

    @property
    def mha_bytes_per_token(self) -> int | None:
        if self.num_query_heads is None:
            return None
        return 2 * self.num_layers * self.num_query_heads * self.head_dim * self.dtype_bytes

    def bytes_for_tokens(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        return self.bytes_per_token * tokens

    def max_tokens_for_budget(self, budget_bytes: int) -> int:
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")
        return budget_bytes // self.bytes_per_token

    def max_concurrent_requests(self, budget_bytes: int, tokens_per_request: int) -> int:
        if tokens_per_request <= 0:
            raise ValueError("tokens_per_request must be positive")
        return budget_bytes // self.bytes_for_tokens(tokens_per_request)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["dtype"] = str(self.dtype)
        data["dtype_bytes"] = self.dtype_bytes
        data["bytes_per_token"] = self.bytes_per_token
        data["gqa_group_size"] = self.gqa_group_size
        data["mha_bytes_per_token"] = self.mha_bytes_per_token
        data["gqa_kv_memory_savings_fraction"] = (
            1 - (self.bytes_per_token / self.mha_bytes_per_token)
            if self.mha_bytes_per_token is not None
            else None
        )
        return data


def observed_kv_cache_bytes(past_key_values: Any) -> int:
    """Return physical KV tensor bytes across common Hugging Face cache representations."""
    tensors: list[torch.Tensor] = []
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        tensors.extend(value for value in past_key_values.key_cache if isinstance(value, torch.Tensor))
        tensors.extend(value for value in past_key_values.value_cache if isinstance(value, torch.Tensor))
    elif isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                tensors.extend(value for value in layer if isinstance(value, torch.Tensor))
    else:
        raise TypeError(f"unsupported KV-cache type: {type(past_key_values).__name__}")
    if not tensors:
        raise ValueError("KV cache contained no observable key/value tensors")
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
