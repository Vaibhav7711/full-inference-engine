import pytest

from engine.metrics.stats import latency_summary_ms, percentile


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == 38.5


def test_summary_has_expected_fields() -> None:
    summary = latency_summary_ms([10.0, 20.0, 30.0])
    assert summary == {"mean": 20.0, "p50": 20.0, "p95": 29.0, "p99": 29.8, "min": 10.0, "max": 30.0}


@pytest.mark.parametrize("samples, percent", [([], 50), ([1.0], -1), ([1.0], 101)])
def test_percentile_rejects_invalid_input(samples: list[float], percent: float) -> None:
    with pytest.raises(ValueError):
        percentile(samples, percent)
