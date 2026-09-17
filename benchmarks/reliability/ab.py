"""Controlled A/B: does anything actually change when one engine setting changes?

The soak reports numbers per configuration, but its configurations differ in several ways
at once, so differences between them cannot be attributed to any single cause. This script
varies exactly one setting and holds everything else identical - same pool, same
concurrency, same prompts, same seeds, same model object - and repeats each arm so a
difference can be compared against run-to-run spread instead of assumed real.

It runs closed-loop by design. Open-loop arrivals above the engine's service rate make
every latency number a measure of queue depth, which is the same for both arms and hides
whatever the setting actually did.

    python -m benchmarks.reliability.ab --setting cuda_graphs --repeats 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.reliability.soak import SoakConfig, run_repeated

# Each arm is (label, engine keyword overrides). Everything not listed is shared.
SETTINGS: dict[str, list[tuple[str, dict]]] = {
    "cuda_graphs": [
        ("graphs_off", {"cuda_graph_batch_sizes": None}),
        ("graphs_on", {"cuda_graph_batch_sizes": (1, 2, 4, 8)}),
    ],
    "prefix_cache": [
        ("cache_off", {"prefix_cache_blocks": 0}),
        ("cache_on", {"prefix_cache_blocks": 64}),
    ],
    "kv_dtype": [
        ("fp16_kv", {"kv_cache_dtype": "fp16"}),
        ("int8_kv", {"kv_cache_dtype": "int8"}),
    ],
}


def _verdict(baseline: dict, variant: dict) -> str:
    """Call a difference real only when it exceeds the noise in both arms."""
    if not baseline.get("n") or not variant.get("n"):
        return "no data"
    base, var = baseline["median"], variant["median"]
    if base == 0:
        return "baseline is zero"
    change = (var - base) / base
    noise = max(baseline["spread"], variant["spread"])
    if abs(change) <= noise:
        return (
            f"unresolved: {change:+.1%} median change is within {noise:.1%} run-to-run spread"
        )
    return f"{change:+.1%} (spread {noise:.1%})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--setting", default="cuda_graphs", choices=sorted(SETTINGS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--max-waiting", type=int, default=64)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default="results/soak_ab.json")
    args = parser.parse_args()

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    # One model, many engines: reloading weights per arm would add minutes and change
    # nothing the comparison is about.
    loaded = load_model(args.model)
    shared = dict(
        num_blocks=args.num_blocks, block_size=16, max_active=args.max_active,
        max_waiting_requests=args.max_waiting, prefix_cache_blocks=64,
    )
    config = SoakConfig(
        duration_s=args.duration, concurrency=args.concurrency, seed=args.seed,
        cancel_probability=0.02,  # low: this measures generation, not cancellation
    )

    arms = {}
    for label, overrides in SETTINGS[args.setting]:
        settings = {**shared, **overrides}

        def make_engine(settings=settings):
            return ContinuousBatchingEngine(
                loaded.model, loaded.tokenizer, loaded.device, **settings
            )

        print(f"\n=== {label} ({args.repeats} runs) ===")
        repeated = run_repeated(make_engine, config, repeats=args.repeats, label=label)
        arms[label] = repeated
        for metric in ("latency.itl_p50", "latency.ttft_p50", "waste_ratio"):
            stats = repeated.summary(metric)
            if stats.get("n"):
                print(f"  {metric:24s} median={stats['median']:.3f}  "
                      f"[{stats['min']:.3f}, {stats['max']:.3f}]  spread={stats['spread']:.1%}")
        if not repeated.ok:
            print(f"  INVARIANT VIOLATIONS: {[v for r in repeated.runs for v in r.violations]}")
        elif repeated.coverage_gaps:
            # Expected here: this configuration is roomy and barely cancels, by design,
            # so that the measurement reflects generation rather than churn.
            print(f"  (coverage gaps, not failures: {len(repeated.coverage_gaps)})")

    labels = [label for label, _ in SETTINGS[args.setting]]
    baseline, variant = arms[labels[0]], arms[labels[1]]
    print(f"\n=== {labels[1]} vs {labels[0]} ===")
    comparison = {}
    for metric in ("latency.itl_p50", "latency.itl_p99", "latency.ttft_p50", "waste_ratio"):
        verdict = _verdict(baseline.summary(metric), variant.summary(metric))
        comparison[metric] = verdict
        print(f"  {metric:24s} {verdict}")

    payload = {
        "setting": args.setting, "repeats": args.repeats,
        "config": {**config.__dict__, **shared},
        "arms": {label: arm.to_dict() for label, arm in arms.items()},
        "comparison": comparison,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {out}")
    return 0 if all(arm.ok for arm in arms.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
