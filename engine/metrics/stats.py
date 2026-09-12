"""Deterministic summary statistics for benchmark samples."""

from __future__ import annotations

from math import floor


def percentile(samples: list[float], percent: float) -> float:
    """Linear-interpolated percentile; requires at least one sample."""
    if not samples:
        raise ValueError("percentile requires at least one sample")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be in [0, 100]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * (percent / 100)
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("latency summary requires at least one sample")
    return {
        "mean": sum(samples) / len(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "min": min(samples),
        "max": max(samples),
    }
