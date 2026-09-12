from types import SimpleNamespace

import pytest
import torch

from engine.cache import KVCacheGeometry, observed_kv_cache_bytes


def test_gqa_geometry_and_capacity_accounting() -> None:
    config = SimpleNamespace(num_hidden_layers=4, num_key_value_heads=2, num_attention_heads=8, hidden_size=64)
    geometry = KVCacheGeometry.from_model_config(config, torch.float16)
    assert geometry.head_dim == 8
    assert geometry.bytes_per_token == 256
    assert geometry.bytes_for_tokens(10) == 2560
    assert geometry.max_tokens_for_budget(1024) == 4
    assert geometry.max_concurrent_requests(4096, 4) == 4


def test_observed_dynamic_cache_bytes() -> None:
    layer = SimpleNamespace(keys=torch.zeros((1, 2, 3, 4), dtype=torch.float16), values=torch.zeros((1, 2, 3, 4), dtype=torch.float16))
    assert observed_kv_cache_bytes(SimpleNamespace(layers=[layer])) == 96


@pytest.mark.parametrize("tokens", [-1])
def test_rejects_invalid_token_count(tokens: int) -> None:
    geometry = KVCacheGeometry(1, 1, 1, torch.float16)
    with pytest.raises(ValueError):
        geometry.bytes_for_tokens(tokens)
