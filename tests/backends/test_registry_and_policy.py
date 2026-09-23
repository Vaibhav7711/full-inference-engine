"""The hooks that make this engine more than "Qwen3 on a T4".

Three properties matter and all are testable without a GPU: a backend that cannot run
here says why rather than being silently skipped, a *named* backend that cannot run
raises instead of falling back, and the per-device policy separates what was measured on
an architecture from what is merely available on it.
"""

from __future__ import annotations

import pytest

from engine.backends import (
    Backend, Geometry, defaults_for, describe, names, register, report, resolve, unregister,
)
from engine.backends.policy import MEASURED
from engine.kernels.device import DeviceProfile


def _profile(sm: int) -> DeviceProfile:
    major, minor = divmod(sm, 10)
    return DeviceProfile(name=f"fake sm_{sm}", capability=(major, minor),
                         total_memory_gb=16.0, multiprocessors=40, l2_cache_mb=4.0)


QWEN = Geometry(num_q_heads=16, num_kv_heads=8, head_dim=128, block_size=16)


@pytest.fixture
def scratch_backend():
    created = []

    def make(name: str, phase: str = "decode", *, reason: str | None = None, priority: int = 0):
        backend = Backend(
            name=name, phase=phase, run=lambda *a, **k: name,
            available=lambda profile, geometry, reason=reason: reason,
            priority=priority, summary=f"test backend {name}",
        )
        register(backend)
        created.append((phase, name))
        return backend

    yield make
    for phase, name in created:
        unregister(phase, name)


def test_builtins_are_registered_for_both_phases() -> None:
    assert {"per_head", "gqa", "split_k", "flash"} <= set(names("decode"))
    assert {"sdpa", "per_token", "tiled", "flash"} <= set(names("prefill"))


def test_describe_reports_a_reason_for_every_unavailable_backend() -> None:
    rows = describe("prefill", _profile(75), QWEN)
    by_name = {row["name"]: row for row in rows}
    assert by_name["tiled"]["available"] is False
    assert "sm_75" in by_name["tiled"]["reason"]
    for row in rows:
        assert row["available"] or row["reason"], f"{row['name']} is unavailable without a reason"


def test_named_backend_that_cannot_run_raises_with_the_reason() -> None:
    with pytest.raises(ValueError, match="tensor cores"):
        resolve("prefill", "tiled", _profile(75), QWEN)


def test_auto_picks_the_highest_priority_available_backend(scratch_backend) -> None:
    scratch_backend("cheap", priority=1)
    scratch_backend("preferred", priority=999)
    assert resolve("decode", "auto", _profile(80), QWEN).name == "preferred"


def test_auto_skips_a_backend_that_cannot_run(scratch_backend) -> None:
    scratch_backend("broken", priority=1000, reason="needs a GPU from 2030")
    scratch_backend("usable", priority=500)
    assert resolve("decode", None, _profile(80), QWEN).name == "usable"


def test_unknown_backend_name_lists_what_exists() -> None:
    with pytest.raises(KeyError, match="per_head"):
        resolve("decode", "does_not_exist", _profile(80), QWEN)


def test_registering_a_duplicate_name_is_refused(scratch_backend) -> None:
    scratch_backend("dupe")
    with pytest.raises(ValueError, match="already registered"):
        register(Backend(name="dupe", phase="decode", run=lambda *a, **k: None))


def test_int8_pool_excludes_the_gather_and_group_backends() -> None:
    int8 = Geometry(16, 8, 128, 16, kv_dtype="int8")
    prefill = {row["name"]: row for row in describe("prefill", _profile(80), int8)}
    decode = {row["name"]: row for row in describe("decode", _profile(80), int8)}
    assert prefill["sdpa"]["available"] is False and "INT8" in prefill["sdpa"]["reason"]
    assert decode["gqa"]["available"] is False
    # per_token is the chunked path that dequantizes in-kernel, so it must survive.
    assert prefill["per_token"]["reason"] is None or "triton" in prefill["per_token"]["reason"]


def test_head_dim_above_the_kernel_limit_is_reported_not_crashed() -> None:
    wide = Geometry(16, 8, 256, 16)
    rows = {row["name"]: row for row in describe("decode", _profile(80), wide)}
    assert rows["per_head"]["available"] is False
    assert "128" in rows["per_head"]["reason"]


def test_measured_defaults_are_used_for_a_measured_architecture() -> None:
    assert 75 in MEASURED, "the T4 results are the engine's reference architecture"
    defaults = defaults_for(_profile(75), QWEN)
    assert defaults.prefill_attention == "sdpa"
    assert defaults.decode_attention == "per_head"
    assert defaults.dtype == "float16"
    assert "T4 A/B" in defaults.reasons["prefill_attention"]


def test_unmeasured_architecture_says_so_and_prefers_bf16() -> None:
    defaults = defaults_for(_profile(89), QWEN)
    assert defaults.dtype == "bfloat16"
    assert "unmeasured" in defaults.reasons["decode_attention"]
    assert "ab.py" in defaults.reasons["measure"]


def test_measured_default_that_cannot_run_here_falls_back_and_says_so() -> None:
    # An INT8 pool on the T4: the measured prefill default (sdpa, which gathers) has no
    # INT8 variant, so the policy must pick another and record why.
    defaults = defaults_for(_profile(75), Geometry(16, 8, 128, 16, kv_dtype="int8"))
    reason = defaults.reasons["prefill_attention"]
    assert "cannot run here" in reason
    # Either an alternative was selected, or none exists and the engine will say so;
    # which of the two depends on whether Triton is installed on this machine.
    assert defaults.prefill_attention != "sdpa" or "no alternative" in reason


def test_report_is_diagnostic_when_nothing_can_run() -> None:
    payload = report(None, Geometry(16, 8, 256, 16))
    assert payload["measured"] is False
    # Still names a default and explains itself rather than refusing to answer.
    assert payload["defaults"]["decode_attention"]
    assert payload["reasons"]
    assert all(not row["available"] for row in payload["decode_backends"])
