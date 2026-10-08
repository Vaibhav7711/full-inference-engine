"""Capability tiers are compute-capability gates, never product names."""

from __future__ import annotations

import pytest

from engine.backends import arch
from engine.kernels.device import DeviceProfile


def _profile(sm: int) -> DeviceProfile:
    major, minor = divmod(sm, 10)
    return DeviceProfile(name=f"sm_{sm}", capability=(major, minor), total_memory_gb=8.0,
                         multiprocessors=24, l2_cache_mb=24.0)


@pytest.mark.parametrize("sm, path", [
    (60, "sm75"), (75, "sm75"), (80, "sm80plus"), (86, "sm80plus"),
    (89, "sm80plus"), (90, "sm80plus"), (100, "sm80plus"), (120, "sm80plus"),
])
def test_path_splits_at_sm80(sm, path):
    assert arch.path_for(_profile(sm)) == path


def test_no_device_is_filed_with_turing():
    assert arch.path_for(None) == "sm75"


def test_gates_are_monotone_and_ordered():
    # Every later tier implies every earlier one; the order is the one the registry's
    # `available()` checks rely on.
    for sm in (75, 80, 86, 89, 90, 100, 120):
        caps = arch.capabilities(_profile(sm))
        if caps["fp4"]:
            assert caps["wgmma"]
        if caps["wgmma"]:
            assert caps["fp8"]
        if caps["fp8"]:
            assert caps["mma_sync"] and caps["async_copy"] and caps["bf16_tensor_cores"]


def test_the_three_cards_this_project_has_touched():
    t4, ada, hopper = (arch.capabilities(_profile(sm)) for sm in (75, 89, 90))
    assert not t4["mma_sync"] and not t4["fp8"]
    assert ada["mma_sync"] and ada["fp8"] and not ada["wgmma"]
    assert hopper["fp8"] and hopper["wgmma"] and not hopper["fp4"]


def test_fp8_gate_agrees_with_the_kv_format_module():
    from engine.kernels.fp8_format import FP8_DTYPE, unavailable_reason

    if FP8_DTYPE is None:
        pytest.skip("torch has no float8_e4m3fn")
    for sm in (75, 86, 89, 90):
        assert (unavailable_reason(_profile(sm)) is None) == arch.has_fp8(sm)
