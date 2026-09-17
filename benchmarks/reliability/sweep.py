"""Run the whole prefill investigation in one GPU session and fit the cost model.

See `docs/experiment-plan-prefill.md` for the pre-registered plan, predictions and decision
rule. This script executes Phase B: three sweeps, one model load, one JSON artefact.

Why a sweep rather than another A/B: the question is not "is 128 better than 512" but "what
are the coefficients of `step_ms = a + b * chunk_tokens`". Two arms give a verdict that can
be null for uninteresting reasons; four points give a line, and the line answers which of
two very different fixes is worth building.

    python -m benchmarks.reliability.sweep
    python -m benchmarks.reliability.sweep --only B1 --repeats 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from benchmarks.reliability.ab import PROMPT_PROFILES, graph_buckets, mean_prompt_tokens
from benchmarks.reliability.soak import SoakConfig, run_repeated


def fit_line(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Least-squares fit of y = a + b*x, with R^2 so a bad fit is visible as a bad fit."""
    n = len(xs)
    if n < 2:
        return {"a": 0.0, "b": 0.0, "r2": 0.0, "n": n}
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return {"a": mean_y, "b": 0.0, "r2": 0.0, "n": n}
    b = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    a = mean_y - b * mean_x
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return {"a": a, "b": b, "r2": 1 - ss_res / ss_tot if ss_tot else 1.0, "n": n}


def _engine_factory(loaded, settings):
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    def make():
        return ContinuousBatchingEngine(
            loaded.model, loaded.tokenizer, loaded.device, **settings
        )

    return make


def _point(loaded, label, settings, config, repeats):
    repeated = run_repeated(_engine_factory(loaded, settings), config, repeats, label)
    step = repeated.summary("step_timing.prefill_step_p50_ms")
    decode = repeated.summary("step_timing.decode_step_p50_ms")
    share = repeated.summary("step_timing.prefill_step_fraction")
    gap = repeated.summary("step_timing.expected_gap_ms")
    ttft = repeated.summary("latency.ttft_p50")
    print(f"  {label:14s} prefill_step={step.get('median', 0):7.2f} ms "
          f"(spread {step.get('spread', 0):5.1%})  decode={decode.get('median', 0):6.2f}  "
          f"share={share.get('median', 0):5.1%}  gap={gap.get('median', 0):6.2f}  "
          f"ttft={ttft.get('median', 0):7.1f}")
    if not repeated.ok:
        print(f"    INVARIANT VIOLATIONS: {[v for r in repeated.runs for v in r.violations]}")
    paths = [
        (r.step_timing, getattr(r, "counts", None)) for r in repeated.runs
    ]
    return {
        "label": label, "settings": {k: v for k, v in settings.items() if k != "model"},
        "prefill_step_p50_ms": step, "decode_step_p50_ms": decode,
        "prefill_step_fraction": share, "expected_gap_ms": gap, "ttft_p50_ms": ttft,
        "ok": repeated.ok,
        "violations": [v for r in repeated.runs for v in r.violations],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--only", nargs="*", default=["B1", "B2", "B3"],
                        choices=["B1", "B2", "B3"])
    parser.add_argument("--out", default="results/prefill_sweep.json")
    args = parser.parse_args()

    from engine.model import load_model

    started = perf_counter()
    loaded = load_model(args.model)
    base = dict(
        num_blocks=args.num_blocks, block_size=16, max_active=args.max_active,
        max_waiting_requests=64, prefix_cache_blocks=0,
        cuda_graph_batch_sizes=graph_buckets(args.max_active),
    )
    # Prefix caching is off throughout: shared prefixes would let some requests skip
    # prefill entirely, which is exactly the work being measured.
    results: dict[str, object] = {"model": args.model, "base_settings": base, "runs": {}}

    if "B1" in args.only:
        profile = PROMPT_PROFILES["long"]
        mean_prompt = mean_prompt_tokens(profile)
        config = SoakConfig(duration_s=args.duration, concurrency=args.concurrency,
                            seed=200, cancel_probability=0.0, max_new_tokens=(32, 64),
                            **profile)
        print(f"\n=== B1 chunk sweep, long profile (~{mean_prompt:.0f} token prompts) ===")
        points = []
        for chunk in (64, 128, 256, 512):
            settings = {**base, "prefill_chunk_size": chunk,
                        "max_prefill_tokens_per_iteration": chunk}
            points.append(_point(loaded, f"chunk_{chunk}", settings, config, args.repeats))
        xs = [p["settings"]["prefill_chunk_size"] for p in points]
        ys = [p["prefill_step_p50_ms"].get("median", 0.0) for p in points]
        fit = fit_line(xs, ys)
        print(f"\n  fit: step_ms = {fit['a']:.2f} + {fit['b']:.5f} * chunk_tokens"
              f"   (R^2={fit['r2']:.3f})")
        print(f"  a = {fit['a']:.2f} ms fixed per invocation")
        print(f"  b = {fit['b']:.5f} ms/token vs ~0.018 ms/token dense compute floor "
              f"= {fit['b']/0.018:.1f}x floor" if fit["b"] > 0 else "  b <= 0")
        verdict = []
        if fit["a"] > 20:
            verdict.append("a>20ms: launch-bound component is dominant -> graph the prefill path")
        if fit["b"] > 0.1:
            verdict.append("b>0.1ms/token: kernel is far off the compute floor -> tiled prefill kernel")
        if not verdict:
            verdict.append("neither threshold met -> profile before designing anything")
        for line in verdict:
            print(f"  DECISION: {line}")
        results["runs"]["B1"] = {"points": points, "fit": fit, "verdict": verdict,
                                 "mean_prompt_tokens": mean_prompt}

    if "B2" in args.only:
        print("\n=== B2 profile sweep, chunk 128 ===")
        points = []
        for name in ("short", "chat", "long"):
            profile = PROMPT_PROFILES[name]
            config = SoakConfig(duration_s=args.duration, concurrency=args.concurrency,
                                seed=300, cancel_probability=0.0, max_new_tokens=(32, 64),
                                **profile)
            settings = {**base, "prefill_chunk_size": 128,
                        "max_prefill_tokens_per_iteration": 128}
            point = _point(loaded, name, settings, config, args.repeats)
            point["mean_prompt_tokens"] = mean_prompt_tokens(profile)
            points.append(point)
        results["runs"]["B2"] = {"points": points}

    if "B3" in args.only:
        profile = PROMPT_PROFILES["chat"]
        config = SoakConfig(duration_s=args.duration, concurrency=args.concurrency,
                            seed=400, cancel_probability=0.0, max_new_tokens=(32, 64),
                            **profile)
        print(f"\n=== B3 budget vs chunk, chat profile "
              f"(~{mean_prompt_tokens(profile):.0f} token prompts) ===")
        points = []
        for label, budget in (("budget_1x", 128), ("budget_4x", 512)):
            settings = {**base, "prefill_chunk_size": 128,
                        "max_prefill_tokens_per_iteration": budget}
            points.append(_point(loaded, label, settings, config, args.repeats))
        one, four = (p["prefill_step_p50_ms"].get("median", 0.0) for p in points)
        noise = max(p["prefill_step_p50_ms"].get("spread", 0.0) for p in points)
        if one > 0:
            # A packed step does four chunks' worth of work, so it must cost more than a
            # single-chunk step. The question is whether it costs less than four of them:
            # that is what tells you the fixed cost amortises across chunks in a step.
            separate = 4 * one
            amortisation = separate / four if four else 0.0
            print(f"\n  one packed step {four:.2f} ms vs four separate steps "
                  f"{separate:.2f} ms  ->  {amortisation:.2f}x cheaper per unit work "
                  f"(spread {noise:.1%})")
            if amortisation > 1 + noise:
                print(f"  a amortises across chunks in a step: raise "
                      f"max_prefill_tokens_per_iteration above prefill_chunk_size")
            else:
                print("  a does not amortise: fixed cost is per chunk, not per step")
            results_b3_extra = {"packed_ms": four, "separate_ms": separate,
                                "amortisation": amortisation}
        results["runs"]["B3"] = {"points": points,
                                 **(results_b3_extra if one > 0 else {})}

    results["wall_s"] = perf_counter() - started
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nTotal {results['wall_s'] / 60:.1f} min. Saved -> {out}")
    violations = [v for run in results["runs"].values()
                  for p in run["points"] for v in p["violations"]]
    if violations:
        print(f"INVARIANT VIOLATIONS across the sweep: {violations[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
