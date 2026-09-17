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
