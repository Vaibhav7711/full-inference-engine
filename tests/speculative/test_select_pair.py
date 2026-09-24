from benchmarks.speculative.select_pair import choose, summarize


def _artifact(model: str, speedups: list[float], acceptance: float) -> dict:
    rows = []
    for index, speedup in enumerate(speedups):
        proposed = 100
        rows.append({
            "depth": 3, "stratum": f"s{index}", "baseline_ms": 100.0,
            "speculative_ms": 100.0 / speedup, "speedup": speedup,
            "proposed": proposed, "accepted": int(proposed * acceptance),
            "draft_ms": 10.0, "tokens_match": True,
        })
    return {"target_model": "target", "draft_model": model, "rows": rows}


def test_summary_aggregates_acceptance_and_wall_cost() -> None:
    summary = summarize(_artifact("draft", [1.1, 1.2], 0.7))[0]
    assert summary["acceptance_rate"] == 0.7
    assert summary["aggregate_speedup"] > 1.1
    assert summary["draft_wall_fraction"] > 0


def test_selection_balances_speed_gate_and_stratum_floor() -> None:
    fast_but_regressive = _artifact("large", [1.5, 0.8], 0.9)
    balanced = _artifact("small", [1.12, 1.08], 0.7)
    winner, _ = choose(
        [fast_but_regressive, balanced], min_speedup=1.05, min_stratum_speedup=0.95,
    )
    assert winner is not None
    assert winner["draft_model"] == "small"


def test_token_mismatch_is_never_eligible() -> None:
    artifact = _artifact("bad", [1.3, 1.3], 0.9)
    artifact["rows"][0]["tokens_match"] = False
    winner, _ = choose([artifact], min_speedup=1.05, min_stratum_speedup=0.95)
    assert winner is None
