"""Device profiling drives kernel defaults, so its logic is tested without a GPU."""

from __future__ import annotations

import engine.kernels.device as device
from engine.kernels.device import DeviceProfile, prefill_tile_defaults

T4 = DeviceProfile("Tesla T4", (7, 5), 15.8, 40, 4.0)
ADA_4060 = DeviceProfile("NVIDIA GeForce RTX 4060", (8, 9), 8.6, 24, 24.0)


def test_compute_capability_is_reported_as_an_integer() -> None:
    assert T4.sm == 75 and ADA_4060.sm == 89


def test_async_copy_is_ampere_and_later_only() -> None:
    # num_stages > 2 requires cp.async; on Turing the extra stages only cost registers.
    assert not T4.supports_async_copy
    assert ADA_4060.supports_async_copy


def test_pipeline_depth_follows_the_architecture() -> None:
    assert T4.prefill_tile_defaults()["num_stages"] == 2
    assert ADA_4060.prefill_tile_defaults()["num_stages"] == 3


def test_defaults_are_turing_safe_when_no_device_is_present() -> None:
    defaults = prefill_tile_defaults()
    assert defaults["num_stages"] == 2, "must not assume cp.async off-GPU"
    assert defaults["block_m"] >= 16 and defaults["block_n"] >= 16


def test_profile_renders_the_facts_that_change_conclusions() -> None:
    rendered = str(ADA_4060)
    for fragment in ("sm_89", "8.6 GB", "24 SMs", "24 MB L2"):
        assert fragment in rendered, rendered


def test_memory_check_is_skipped_without_a_device(monkeypatch) -> None:
    monkeypatch.setattr(device, "current_device", lambda: None)
    fits, reason = device.fits_in_memory(8_000_000_000)
    assert fits and "no CUDA device" in reason
