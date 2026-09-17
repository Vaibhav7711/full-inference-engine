"""The A/B harness must not fail deep inside a repeat loop over its own configuration."""

from __future__ import annotations

from benchmarks.reliability.ab import SETTINGS, _clamp_graph_buckets, graph_buckets


def test_graph_buckets_never_exceed_max_active() -> None:
    # The engine rejects any bucket above max_active, at construction time.
    for max_active in (1, 2, 3, 4, 8, 16, 100):
        buckets = graph_buckets(max_active)
        assert buckets, "at least one bucket is always needed"
        assert all(1 <= size <= max_active for size in buckets), (max_active, buckets)
        assert list(buckets) == sorted(buckets)


def test_clamp_drops_oversized_buckets_and_leaves_other_settings_alone() -> None:
    clamped = _clamp_graph_buckets(
        {"cuda_graph_batch_sizes": (1, 2, 4, 8, 16), "num_blocks": 256}, max_active=8
    )
    assert clamped["cuda_graph_batch_sizes"] == (1, 2, 4, 8)
    assert clamped["num_blocks"] == 256


def test_clamp_removes_the_key_entirely_when_nothing_fits() -> None:
    clamped = _clamp_graph_buckets({"cuda_graph_batch_sizes": (16, 32)}, max_active=8)
    assert "cuda_graph_batch_sizes" not in clamped


def test_clamp_passes_through_an_explicit_none() -> None:
    assert _clamp_graph_buckets({"cuda_graph_batch_sizes": None}, 8) == {
        "cuda_graph_batch_sizes": None
    }


def test_every_declared_setting_survives_clamping_at_common_concurrencies() -> None:
    for name, arms in SETTINGS.items():
        for max_active in (1, 4, 8, 16):
            for label, overrides in arms:
                settings = _clamp_graph_buckets(dict(overrides), max_active)
                buckets = settings.get("cuda_graph_batch_sizes")
                assert buckets is None or all(s <= max_active for s in buckets), (
                    name, label, max_active, buckets
                )


# ------------------------------------------------ pre-flight: can the treatment take effect?
from benchmarks.reliability.ab import (  # noqa: E402
    PROMPT_PROFILES, binding_check, mean_prompt_tokens,
)


def test_chunk_512_versus_128_is_non_binding_on_short_prompts() -> None:
    """The exact run that was wasted: both arms fit a ~122-token prompt in one chunk."""
    mean_prompt = mean_prompt_tokens(PROMPT_PROFILES["short"])
    assert 100 < mean_prompt < 140
    problem = binding_check(SETTINGS["prefill_chunk_large"], mean_prompt)
    assert problem is not None and "non-binding" in problem
    assert "chunk 128 -> 1 chunk(s)" in problem and "chunk 512 -> 1 chunk(s)" in problem


def test_chunk_512_versus_128_binds_once_prompts_exceed_the_chunk() -> None:
    for name in ("chat", "long"):
        mean_prompt = mean_prompt_tokens(PROMPT_PROFILES[name])
        assert binding_check(SETTINGS["prefill_chunk_large"], mean_prompt) is None, name


def test_chunk_32_versus_128_binds_even_on_short_prompts() -> None:
    """This treatment was real, which is why its finding stands."""
    mean_prompt = mean_prompt_tokens(PROMPT_PROFILES["short"])
    assert binding_check(SETTINGS["prefill_chunk"], mean_prompt) is None


def test_settings_without_chunk_sizes_are_not_judged_by_this_check() -> None:
    # The check covers chunk-size binding only. It does not and cannot detect every
    # null-by-construction design - the p999 CUDA-graph comparison was a different kind.
    mean_prompt = mean_prompt_tokens(PROMPT_PROFILES["short"])
    for setting in ("cuda_graphs", "prefix_cache", "kv_dtype"):
        assert binding_check(SETTINGS[setting], mean_prompt) is None


def test_prompt_profiles_are_ordered_and_plausible() -> None:
    means = [mean_prompt_tokens(PROMPT_PROFILES[n]) for n in ("short", "chat", "long")]
    assert means == sorted(means)
    # Real chat traffic carries a system prompt plus history; the default profile does not.
    assert means[0] < 200 and means[1] > 500 and means[2] > 1500
