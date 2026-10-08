"""The attention kernel chosen for speculative verification, per device.

Verification attends `depth + 1` queries against a long paged prefix. Before this field
existed it ran on whatever `prefill_attention` resolved to - SDPA's prefix gather on the
T4, eager FlashAttention on Ada - so these tests pin the rule that picks an in-place
paged kernel instead, and that `"prefill"` remains selectable as the A/B baseline.

Pure-function tests pass Triton or not; the end-to-end one only asserts consistency
with what `describe()` says can run here.
"""

from __future__ import annotations

from engine.backends.policy import DeviceDefaults, _verify_default, defaults_for
from engine.backends.registry import Geometry, describe
from engine.kernels.device import DeviceProfile

QWEN = Geometry(num_q_heads=16, num_kv_heads=8, head_dim=128, block_size=16)


def _profile(sm: int) -> DeviceProfile:
    major, minor = divmod(sm, 10)
    return DeviceProfile(name=f"sm_{sm}", capability=(major, minor), total_memory_gb=16.0,
                         multiprocessors=40, l2_cache_mb=4.0)


def test_tiled_tile_is_preferred_from_sm80_when_runnable():
    name, reason = _verify_default(_profile(89), {"sdpa", "per_token", "tiled", "flash"})
    assert name == "tiled"
    assert "block_m=16" in reason and "unmeasured" in reason


def test_turing_never_picks_tiled_even_when_listed_available():
    # On sm_75 `tl.dot` lowers to FMA (journal: "never used the tensor cores"); the policy
    # must not route verification there regardless of what the registry reports.
    name, _ = _verify_default(_profile(75), {"sdpa", "per_token", "tiled"})
    assert name == "per_token"


def test_falls_back_to_the_prefill_backend_when_no_in_place_kernel_runs():
    name, reason = _verify_default(_profile(89), {"sdpa", "flash"})
    assert name == "prefill"
    assert "no in-place paged kernel" in reason


def test_no_device_behaves_like_the_lowest_capability():
    assert _verify_default(None, {"per_token", "tiled"})[0] == "per_token"


def test_device_defaults_keep_prefill_as_the_baseline_default():
    # Existing constructions pass three or four positional fields; the new one must not
    # change their meaning or make `"prefill"` unreachable.
    defaults = DeviceDefaults("per_head", "sdpa", "float16")
    assert defaults.verify_attention == "prefill"


def test_defaults_for_reports_a_verify_choice_consistent_with_availability():
    for sm in (75, 89):
        defaults = defaults_for(_profile(sm), QWEN)
        available = {row["name"] for row in describe("prefill", _profile(sm), QWEN)
                     if row["available"]}
        assert defaults.verify_attention in {"tiled", "per_token", "prefill"}
        if defaults.verify_attention != "prefill":
            assert defaults.verify_attention in available
        if sm == 75:
            assert defaults.verify_attention != "tiled"
        assert "verify_attention" in defaults.reasons
