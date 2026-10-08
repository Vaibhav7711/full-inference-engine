"""An FP8 KV pool is a geometry the backend table must describe honestly.

A scaled pool is served by its own dequantising kernel pair, so every backend that reads
fp16 pages directly has to report itself unavailable *with the storage type in the
reason*, exactly as it already does for INT8. These run without Triton: `describe()` only
asks each backend whether it could run, and the refusal strings are what is under test.
"""

from __future__ import annotations

from engine.backends.policy import defaults_for
from engine.backends.registry import Geometry, describe
from engine.kernels.device import DeviceProfile

FP16 = Geometry(num_q_heads=16, num_kv_heads=8, head_dim=128, block_size=16)
FP8 = Geometry(num_q_heads=16, num_kv_heads=8, head_dim=128, block_size=16, kv_dtype="fp8")
INT8 = Geometry(num_q_heads=16, num_kv_heads=8, head_dim=128, block_size=16, kv_dtype="int8")
ADA = DeviceProfile(name="RTX 4060", capability=(8, 9), total_memory_gb=8.0,
                    multiprocessors=24, l2_cache_mb=24.0)


def _table(phase, geometry):
    return {row["name"]: row for row in describe(phase, ADA, geometry)}


def test_fp16_only_prefill_backends_refuse_an_fp8_pool_and_name_the_type():
    rows = _table("prefill", FP8)
    for name in ("sdpa", "tiled", "flash_dense"):
        assert not rows[name]["available"], name
        assert "FP8" in rows[name]["reason"], (name, rows[name]["reason"])


def test_fp16_only_decode_backends_refuse_an_fp8_pool_and_name_the_type():
    rows = _table("decode", FP8)
    for name in ("gqa", "split_k"):
        assert not rows[name]["available"], name
        assert "FP8" in rows[name]["reason"], (name, rows[name]["reason"])


def test_the_dequantising_names_are_gated_the_same_way_as_for_fp16():
    # `per_head` and `per_token` are the names the engine reports for a scaled pool;
    # whether they can run depends only on Triton and head geometry, never on kv_dtype.
    for phase, name in (("decode", "per_head"), ("prefill", "per_token")):
        assert _table(phase, FP8)[name]["available"] == _table(phase, FP16)[name]["available"]
        assert _table(phase, FP8)[name]["available"] == _table(phase, INT8)[name]["available"]


def test_int8_reasons_still_say_int8():
    assert "INT8" in _table("prefill", INT8)["sdpa"]["reason"]
    assert "FP8" not in _table("prefill", INT8)["sdpa"]["reason"]


def test_policy_defaults_resolve_for_an_fp8_pool_without_raising():
    defaults = defaults_for(ADA, FP8)
    assert defaults.decode_attention
    assert defaults.prefill_attention
    # The measured Ada prefill default is `flash`, which reads fp16 pages; with a scaled
    # pool the policy must have replaced it and said so, or kept it with the refusal
    # reason recorded - either is honest, silence is not.
    assert "prefill_attention" in defaults.reasons
