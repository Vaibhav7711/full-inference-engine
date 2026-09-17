"""The sweep's model fit must be trustworthy before a decision rule is hung on it."""

from __future__ import annotations

import pytest

from benchmarks.reliability.sweep import fit_line


def test_fit_recovers_a_known_line_exactly() -> None:
    xs = [64.0, 128.0, 256.0, 512.0]
    a, b = 30.0, 0.04
    fit = fit_line(xs, [a + b * x for x in xs])
    assert fit["a"] == pytest.approx(a)
    assert fit["b"] == pytest.approx(b)
    assert fit["r2"] == pytest.approx(1.0)


def test_flat_data_reports_zero_slope_and_the_fixed_cost() -> None:
    """The launch-bound outcome: cost independent of chunk size."""
    fit = fit_line([64.0, 128.0, 256.0, 512.0], [42.0, 42.0, 42.0, 42.0])
    assert fit["b"] == pytest.approx(0.0)
    assert fit["a"] == pytest.approx(42.0)


def test_pure_compute_data_reports_near_zero_intercept() -> None:
    """The compute-bound outcome: cost proportional to tokens."""
    fit = fit_line([64.0, 128.0, 256.0, 512.0], [0.02 * x for x in [64, 128, 256, 512]])
    assert fit["a"] == pytest.approx(0.0, abs=1e-9)
    assert fit["b"] == pytest.approx(0.02)


def test_noisy_data_lowers_r2_so_a_bad_fit_is_visible() -> None:
    fit = fit_line([64.0, 128.0, 256.0, 512.0], [42.0, 10.0, 60.0, 25.0])
    assert fit["r2"] < 0.5, "scattered points must not be reported as a confident line"


def test_degenerate_inputs_do_not_raise() -> None:
    assert fit_line([], [])["n"] == 0
    assert fit_line([128.0], [42.0])["n"] == 1
    assert fit_line([128.0, 128.0], [42.0, 43.0])["b"] == 0.0


def test_decision_thresholds_separate_the_two_designs() -> None:
    """The Phase C rule must classify each hypothesis correctly from a fitted line."""
    launch_bound = fit_line([64.0, 128.0, 256.0, 512.0], [41.0, 42.0, 42.5, 44.0])
    assert launch_bound["a"] > 20 and launch_bound["b"] < 0.05

    compute_bound = fit_line([64.0, 128.0, 256.0, 512.0],
                             [10.0, 18.0, 34.0, 66.0])
    assert compute_bound["b"] > 0.1
