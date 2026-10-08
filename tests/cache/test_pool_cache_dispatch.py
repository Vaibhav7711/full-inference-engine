"""The prefill cache adapter picks its writer from the pool's storage dtype.

Found on the first Ada A/B run: `BatchedPoolBackedPrefillCache` dispatched on "is there a
scale pool" and assumed INT8, so an FP8 pool was handed to the INT8 writer, which refused
on dtype and killed `kv_dtype_all` 15 minutes in. This pins the dispatch without Triton by
stubbing the two kernel modules and recording which one was called.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from engine.cache.pool_cache import BatchedPoolBackedPrefillCache

FP8 = getattr(torch, "float8_e4m3fn", None)


class _Recorder:
    def __init__(self):
        self.calls: list[str] = []

    def stub(self, name):
        def write(*args, **kwargs):
            self.calls.append(name)
        return write


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    kv_write = types.ModuleType("engine.kernels.kv_write")
    kv_write.write_prefill_kv_batched = rec.stub("fp16")
    int8 = types.ModuleType("engine.kernels.int8_paged_kv")
    int8.write_prefill_int8_kv_batched = rec.stub("int8")
    fp8 = types.ModuleType("engine.kernels.fp8_paged_kv")
    fp8.write_prefill_fp8_kv_batched = rec.stub("fp8")
    for module in (kv_write, int8, fp8):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return rec


def _cache(dtype, scaled: bool):
    layers = 2
    pools = [torch.zeros(4, 16, 8, 128, dtype=dtype) for _ in range(layers)]
    scales = [torch.zeros(4, 16, 8, dtype=torch.float16) for _ in range(layers)] if scaled else None
    return BatchedPoolBackedPrefillCache(
        pools, [p.clone() for p in pools], torch.zeros(1, 4, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32), padded_length=8,
        key_scale_pool=scales, value_scale_pool=scales,
    )


def test_fp16_pool_uses_the_plain_writer(recorder):
    cache = _cache(torch.float16, scaled=False)
    k = torch.zeros(1, 8, 8, 128, dtype=torch.float16)
    cache.update(k, k, 0)
    assert recorder.calls == ["fp16"]


def test_int8_pool_uses_the_int8_writer(recorder):
    cache = _cache(torch.int8, scaled=True)
    k = torch.zeros(1, 8, 8, 128, dtype=torch.float16)
    cache.update(k, k, 1)
    assert recorder.calls == ["int8"]


@pytest.mark.skipif(FP8 is None, reason="torch has no float8_e4m3fn")
def test_fp8_pool_uses_the_fp8_writer_not_the_int8_one(recorder):
    cache = _cache(FP8, scaled=True)
    k = torch.zeros(1, 8, 8, 128, dtype=torch.float16)
    cache.update(k, k, 0)
    assert recorder.calls == ["fp8"]


def test_one_write_per_layer_is_still_enforced(recorder):
    cache = _cache(torch.float16, scaled=False)
    k = torch.zeros(1, 8, 8, 128, dtype=torch.float16)
    cache.update(k, k, 0)
    with pytest.raises(RuntimeError, match="exactly one write"):
        cache.update(k, k, 0)
