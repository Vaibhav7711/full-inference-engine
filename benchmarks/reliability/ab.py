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
    # Graph buckets are clamped to max_active at build time, so this list is an upper
    # bound rather than a fixed configuration.
    "cuda_graphs": [
        ("graphs_off", {"cuda_graph_batch_sizes": None}),
        ("graphs_on", {"cuda_graph_batch_sizes": (1, 2, 4, 8, 16, 32, 64)}),
    ],
    "prefix_cache": [
        ("cache_off", {"prefix_cache_blocks": 0}),
        ("cache_on", {"prefix_cache_blocks": 64}),
    ],
    "kv_dtype": [
        ("fp16_kv", {"kv_cache_dtype": "fp16"}),
        ("int8_kv", {"kv_cache_dtype": "int8"}),
    ],
    # Smaller chunks mean a prefill-carrying step interrupts decoding for less time, at
    # the cost of spreading a prompt over more steps and so delaying its first token.
    # This is the ITL-versus-TTFT trade that any QoS policy has to choose a point on.
    "prefill_chunk": [
        ("chunk_128", {"prefill_chunk_size": 128,
                       "max_prefill_tokens_per_iteration": 128}),
        ("chunk_32", {"prefill_chunk_size": 32,
                      "max_prefill_tokens_per_iteration": 32}),
    ],
    # If a prefill step costs the same at 128 and 512 tokens, its cost is per-invocation
    # overhead rather than work, and no scheduling change can reduce it.
    "prefill_chunk_large": [
        ("chunk_128", {"prefill_chunk_size": 128,
                       "max_prefill_tokens_per_iteration": 128}),
        ("chunk_512", {"prefill_chunk_size": 512,
                       "max_prefill_tokens_per_iteration": 512}),
    ],
    "prefill_chunk_small": [
        ("chunk_128", {"prefill_chunk_size": 128,
                       "max_prefill_tokens_per_iteration": 128}),
        ("chunk_16", {"prefill_chunk_size": 16,
                      "max_prefill_tokens_per_iteration": 16}),
    ],
}


# Prompt-length profiles. The default soak workload is far shorter than real chat
# traffic, and prompt length decides whether a prefill setting has any effect at all.
PROMPT_PROFILES: dict[str, dict] = {
    "short": {"prefix_tokens": (16, 128), "suffix_tokens": (4, 96)},      # ~20-224
    "chat": {"prefix_tokens": (256, 768), "suffix_tokens": (32, 256)},    # ~288-1024
    "long": {"prefix_tokens": (1024, 2048), "suffix_tokens": (64, 512)},  # ~1088-2560
}


def mean_prompt_tokens(profile: dict) -> float:
    low_p, high_p = profile["prefix_tokens"]
    low_s, high_s = profile["suffix_tokens"]
    return (low_p + high_p) / 2 + (low_s + high_s) / 2


def binding_check(arms: list[tuple[str, dict]], mean_prompt: float) -> str | None:
    """Refuse an experiment whose varied setting cannot take effect on this workload.

    Two runs have already been wasted on treatments that were identical by construction:
    a CUDA-graph arm compared against itself on the un-graphed prefill path, and chunk 512
    versus chunk 128 on prompts averaging 122 tokens, where both fit in a single chunk. A
    null result from a non-binding treatment looks exactly like a null result from a real
    one, which is what makes it expensive.
    """
    chunk_sizes = [
        overrides.get("prefill_chunk_size") for _, overrides in arms
        if overrides.get("prefill_chunk_size") is not None
    ]
    if len(chunk_sizes) < 2:
        return None
    chunks_per_prompt = {size: max(1, -(-int(mean_prompt) // size)) for size in chunk_sizes}
    if len(set(chunks_per_prompt.values())) == 1:
        counts = ", ".join(f"chunk {k} -> {v} chunk(s)" for k, v in chunks_per_prompt.items())
        return (
            f"non-binding: at a mean prompt of {mean_prompt:.0f} tokens every arm needs the "
            f"same number of chunks ({counts}), so the arms are identical treatments. "
            f"Use --prompt-profile chat or long, or pick chunk sizes below {mean_prompt:.0f}."
        )
    return None


def graph_buckets(max_active: int) -> tuple[int, ...]:
    """Powers of two up to `max_active`, which the engine requires buckets to respect.

    Hard-coding a bucket list couples the benchmark to one concurrency setting; the engine
    rejects any bucket above `max_active`, and it does so at construction time, several
    frames inside the repeat loop where the message is hard to place.
    """
    buckets = tuple(size for size in (1, 2, 4, 8, 16, 32, 64) if size <= max_active)
    return buckets or (1,)


def _clamp_graph_buckets(settings: dict, max_active: int) -> dict:
    """Keep any explicitly requested buckets inside the engine's contract."""
    requested = settings.get("cuda_graph_batch_sizes")
    if requested is None:
        return settings
    kept = tuple(size for size in requested if 1 <= size <= max_active)
    settings = dict(settings)
    if kept:
        settings["cuda_graph_batch_sizes"] = kept
    else:
        settings.pop("cuda_graph_batch_sizes")
    return settings


def _verdict(baseline: dict, variant: dict) -> str:  # noqa: D401
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
    parser.add_argument("--prompt-profile", default="short", choices=sorted(PROMPT_PROFILES),
                        help="prompt length distribution; prefill settings only bind when "
                             "prompts exceed the chunk size")
    parser.add_argument("--allow-non-binding", action="store_true",
                        help="run even when the varied setting cannot take effect")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="enable graphs in both arms; required to study prefill, "
                             "since an ungraphed decode path swamps the effect")
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
    if args.cuda_graphs:
        shared["cuda_graph_batch_sizes"] = graph_buckets(args.max_active)
    profile = PROMPT_PROFILES[args.prompt_profile]
    mean_prompt = mean_prompt_tokens(profile)
    problem = binding_check(SETTINGS[args.setting], mean_prompt)
    if problem and not args.allow_non_binding:
        print(f"refusing to run: {problem}")
        return 2
    if problem:
        print(f"WARNING {problem}")
    config = SoakConfig(
        duration_s=args.duration, concurrency=args.concurrency, seed=args.seed,
        cancel_probability=0.02,  # low: this measures generation, not cancellation
        **profile,
    )
    print(f"workload: {args.prompt_profile} profile, mean prompt ~{mean_prompt:.0f} tokens")

    arms = {}
    for label, overrides in SETTINGS[args.setting]:
        settings = _clamp_graph_buckets({**shared, **overrides}, args.max_active)

        def make_engine(settings=settings):
            return ContinuousBatchingEngine(
                loaded.model, loaded.tokenizer, loaded.device, **settings
            )

        print(f"\n=== {label} ({args.repeats} runs) ===")
        repeated = run_repeated(make_engine, config, repeats=args.repeats, label=label)
        arms[label] = repeated
        for metric in ("step_timing.expected_gap_ms", "latency.itl_p50",
                       "latency.ttft_p50", "waste_ratio",
                       "mean_decode_batch", "mean_context_tokens",
                       "step_timing.decode_step_p50_ms", "step_timing.prefill_step_p50_ms",
                       "step_timing.prefill_step_fraction"):
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

    for label, arm in arms.items():
        batch = arm.summary("mean_decode_batch").get("median", 0)
        context = arm.summary("mean_context_tokens").get("median", 0)
        print(f"\n{label} operating point: decode batch ~{batch:.1f}, "
              f"context ~{context:.0f} tokens")
        print(f"  compare its ITL against the roofline cell nearest that point:")
        print(f"  python -m benchmarks.kernels.roofline --measured-itl-ms "
              f"{arm.summary('latency.itl_p50').get('median', 0):.2f} "
              f"--itl-batch {max(1, round(batch))} --itl-context {max(1, round(context))}")

    labels = [label for label, _ in SETTINGS[args.setting]]
    baseline, variant = arms[labels[0]], arms[labels[1]]
    print(f"\n=== {labels[1]} vs {labels[0]} ===")
    comparison = {}
    # expected_gap_ms leads because it is the only latency metric sensitive to a change
    # in the *mix* of step kinds. A median over token gaps is not: when prefill rose from
    # 22.7% to 46.9% of steps, itl_p50 stayed inside noise because the median gap was
    # still a decode step, while the average gap worsened 49%.
    for metric in ("step_timing.expected_gap_ms", "step_timing.prefill_share_of_gap",
                   "latency.itl_p50", "latency.itl_p99", "latency.itl_p999",
                   "latency.ttft_p50", "step_timing.decode_step_p50_ms",
                   "step_timing.prefill_step_p50_ms",
                   "step_timing.prefill_step_fraction",
                   "step_timing.prefill_penalty_p50_ms", "waste_ratio"):
        verdict = _verdict(baseline.summary(metric), variant.summary(metric))
        comparison[metric] = verdict
        print(f"  {metric:24s} {verdict}")

    payload = {
        "setting": args.setting, "repeats": args.repeats,
        "config": {**config.__dict__, **shared},
        "prompt_profile": args.prompt_profile,
        "mean_prompt_tokens": mean_prompt,
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
