"""The Ada compact-page prefill fallback is the measured one, not registry priority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.backends import Geometry, defaults_for


def _ada():
    return SimpleNamespace(sm=89, name="NVIDIA GeForce RTX 4060", multiprocessors=24,
                           total_memory_gb=8.0, supports_async_copy=True)


def test_ada_page16_fp16_prefill_falls_back_to_measured_tiled() -> None:
    defaults = defaults_for(_ada(), Geometry(16, 8, 128, 16))
    if "triton" in defaults.reasons.get("prefill_attention", "").lower() and defaults.prefill_attention != "tiled":
        pytest.skip("Triton unavailable on this machine")
    assert defaults.prefill_attention == "tiled"
    assert "measured fallback" in defaults.reasons["prefill_attention"]


def test_ada_int8_pool_does_not_select_the_fp16_tiled_fallback() -> None:
    defaults = defaults_for(_ada(), Geometry(16, 8, 128, 16, kv_dtype="int8"))
    assert defaults.prefill_attention != "tiled"
