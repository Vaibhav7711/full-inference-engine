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

from benchmarks.common import device_clock_record, environment_record, git_record
from benchmarks.reliability.soak import SoakConfig, run_interleaved

def _padded_buckets(max_active: int) -> tuple[int, ...]:
    return tuple(size for size in (1, 2, 4, 8, 16, 32, 64) if size <= max_active) or (1,)


# Each arm is (label, engine keyword overrides). Everything not listed is shared. An
# override may also be a callable of `max_active`, for settings whose value only makes
# sense relative to the engine's width (graph buckets are rejected above `max_active`).
# Two keys are consumed by this harness rather than the engine: `warmup` (False skips
# the pre-timing warmup so first-use capture/JIT lands inside the measured window).
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
    # Chunked-prefill attention implementations, baseline first. `sdpa` gathers the
    # prefix pages and calls torch SDPA (tensor cores on sm_75); `tiled` is the Triton
    # FlashAttention structure, which on the T4 compiles to FMA and spills - kept as an
    # arm so the same run measures it on any GPU where tl.dot does reach the MMA units.
    "prefill_kernel": [
        ("per_token", {"prefill_attention": "per_token"}),
        ("sdpa", {"prefill_attention": "sdpa"}),
        ("tiled", {"prefill_attention": "tiled"}),
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
    # Elementwise fusions, one at a time. Run each twice, with and without
    # --cuda-graphs: they mostly removed launch cost, which graphs remove as well, so
    # their marginal value depends on whether graphs are present.
    "triton_rmsnorm": [
        ("stock_rmsnorm", {"triton_rmsnorm": False}),
        ("triton_rmsnorm", {"triton_rmsnorm": True}),
    ],
    "triton_rope": [
        ("stock_rope", {"triton_rope": False}),
        ("triton_rope", {"triton_rope": True}),
    ],
    "triton_swiglu": [
        ("stock_swiglu", {"triton_swiglu": False}),
        ("triton_swiglu", {"triton_swiglu": True}),
    ],
    # Phase 9 re-run: the original "neutral" result paid two .contiguous() copies per
    # layer that the strided SwiGLU kernel no longer needs.
    "mlp_gate_up": [
        ("separate_gate_up", {"fuse_mlp_gate_up": False}),
        ("fused_gate_up", {"fuse_mlp_gate_up": True}),
    ],
    # Phase 7 vs Phase 8: one exact-width bucket against padded power-of-two buckets.
    "graph_buckets_padded": [
        ("exact_bucket", lambda max_active: {"cuda_graph_batch_sizes": (max_active,)}),
        ("padded_buckets", lambda max_active: {"cuda_graph_batch_sizes": _padded_buckets(max_active)}),
    ],
    # Serving warmup: does paying capture/JIT before the window move the tail?
    "warmup": [
        ("cold_start", {"warmup": False}),
        ("warmed", {"warmup": True}),
    ],
}


# Leave-one-out: everything on, then each optimization removed alone. `full` is the
# recommended serving configuration; contribution is metric(full) - metric(minus_x).
# Persistent metadata, batched KV writes and the like have no toggle and are measured
# by the commit ladder instead (docs/t4-reevaluation-plan.md, session B).
FULL: dict = {
    "cuda_graph_batch_sizes": _padded_buckets,
    "prefix_cache_blocks": 64,
    "prefill_attention": "per_token",
    "triton_rmsnorm": True, "triton_rope": True, "triton_swiglu": True,
}
LEAVE_ONE_OUT: dict[str, dict] = {
    "graphs": {"cuda_graph_batch_sizes": None},
    "prefix_cache": {"prefix_cache_blocks": 0},
    "rmsnorm": {"triton_rmsnorm": False},
    "rope": {"triton_rope": False},
    "swiglu": {"triton_swiglu": False},
}
for _name, _off in LEAVE_ONE_OUT.items():
    SETTINGS[f"loo_{_name}"] = [("full", FULL), (f"minus_{_name}", {**FULL, **_off})]
# All leave-one-out arms interleaved in one run, sharing one `full` baseline.
SETTINGS["loo_all"] = [("full", FULL)] + [
    (f"minus_{name}", {**FULL, **off}) for name, off in LEAVE_ONE_OUT.items()
]


def resolve_arms(arms: list[tuple[str, dict]], max_active: int) -> list[tuple[str, dict]]:
    """Evaluate callable overrides against the engine width."""
    resolved = []
    for label, overrides in arms:
        if callable(overrides):
            overrides = overrides(max_active)
        overrides = {k: (v(max_active) if callable(v) else v) for k, v in overrides.items()}
        resolved.append((label, overrides))
    return resolved


# Settings whose arms are allowed to produce different greedy tokens: they swap a kernel,
# and fp16 kernels with a different reduction order are not bit-identical, so a late
# near-tie can flip. Everything else must be token-identical across arms: a speedup that
# changes output is a bug. Kernel-swapping arms are still gated below: each must agree
# with stock Transformers for the first `--min-identical-tokens` of every prompt, which a
# wrong kernel fails immediately and a rounding difference does not.
TOKEN_DRIFT_EXPECTED = {"kv_dtype", "prefill_kernel", "triton_rmsnorm", "triton_rope",
                        "triton_swiglu", "mlp_gate_up"}


def drift_expected(setting: str) -> bool:
    return setting in TOKEN_DRIFT_EXPECTED or setting.startswith("loo_")

# Fixed prompts for the cross-arm token-identity gate. Long enough to cross a prefill
# chunk and the 128-token decode regime boundary once generation is included.
IDENTITY_PROMPTS = [
    "Explain how a paged KV cache differs from a contiguous one, in detail. " * 6,
    "List ten facts about the number seven.",
    "Write a short story about a lighthouse keeper who " + "kept a very long diary, " * 12,
    "Why?",
]


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


def _stock_reference(loaded, prompts: list[str], max_new_tokens: int) -> list[list[int]]:
    """Greedy continuations from unpatched Transformers: SDPA, DynamicCache, stock RoPE.

    Must run before any engine is built on this model object, since the engine installs
    Triton RMSNorm/SwiGLU on it in place; the RoPE install is process-global and is
    undone here explicitly in case an engine already exists in this process.
    """
    import torch

    from engine.kernels.rope import stock_rope

    model, tokenizer = loaded.model, loaded.tokenizer
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    outputs = []
    with stock_rope(), torch.inference_mode():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(loaded.device)
            generated = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs.append(generated[0, ids.shape[1]:].tolist())
    return outputs


def _first_divergence(arm: list[list[int]], reference: list[list[int]]) -> list[int | None]:
    """Per prompt, the index of the first token that differs from the reference, or None."""
    result = []
    for produced, expected in zip(arm, reference):
        index = next(
            (i for i, (a, b) in enumerate(zip(produced, expected)) if a != b), None,
        )
        if index is None and len(produced) != len(expected):
            index = min(len(produced), len(expected))
        result.append(index)
    return result


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
    parser.add_argument("--no-instrument", action="store_true",
                        help="skip the engine's per-phase step split (host/GPU/sync)")
    parser.add_argument("--allow-token-drift", action="store_true",
                        help="continue even if arms produce different greedy tokens")
    parser.add_argument("--min-identical-tokens", type=int, default=8,
                        help="every arm must match stock Transformers for at least this "
                             "many leading tokens of every identity prompt")
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
    arms_spec = resolve_arms(SETTINGS[args.setting], args.max_active)
    problem = binding_check(arms_spec, mean_prompt)
    if problem and not args.allow_non_binding:
        print(f"refusing to run: {problem}")
        return 2
    if problem:
        print(f"WARNING {problem}")
    config = SoakConfig(
        duration_s=args.duration, concurrency=args.concurrency, seed=args.seed,
        cancel_probability=0.02,  # low: this measures generation, not cancellation
        instrument=not args.no_instrument,
        **profile,
    )
    print(f"workload: {args.prompt_profile} profile, mean prompt ~{mean_prompt:.0f} tokens")

    makers = {}
    arm_settings = {}
    for label, overrides in arms_spec:
        settings = _clamp_graph_buckets({**shared, **overrides}, args.max_active)
        if settings.get("cuda_graph_batch_sizes") is None:
            settings.pop("cuda_graph_batch_sizes", None)
        arm_settings[label] = settings
        harness_keys = {"warmup"}
        engine_kwargs = {k: v for k, v in settings.items() if k not in harness_keys}
        skip_warmup = settings.get("warmup", True) is False

        def make_engine(engine_kwargs=engine_kwargs, skip_warmup=skip_warmup):
            engine = ContinuousBatchingEngine(
                loaded.model, loaded.tokenizer, loaded.device, **engine_kwargs
            )
            engine.skip_benchmark_warmup = skip_warmup
            return engine

        makers[label] = make_engine

    # Stock greedy reference, computed before any engine patches the shared model. It
    # decides which arm is closer to the truth when the gate below finds a difference.
    reference = _stock_reference(loaded, IDENTITY_PROMPTS, 48)

    # Token-identity gate before anything is timed. Each arm generates the same fixed
    # prompts greedily on a warmed engine; the outputs must match unless the setting is
    # one that legitimately changes numerics.
    identity = {}
    for label, make_engine in makers.items():
        engine = make_engine()
        engine.warmup()  # the gate checks tokens, not first-use cost; always warm here
        identity[label] = engine.generate(IDENTITY_PROMPTS, max_new_tokens=48)
        del engine
    labels = list(makers)
    divergence = {label: _first_divergence(identity[label], reference) for label in labels}
    for label in labels:
        print(f"vs stock reference, first divergent token per prompt: {label}={divergence[label]}")
    early = {label: [i for i in divergence[label] if i is not None and i < args.min_identical_tokens]
             for label in labels}
    broken = [label for label in labels if early[label]]
    if broken and not args.allow_token_drift:
        print(f"refusing to run: {broken} diverge from stock Transformers within the first "
              f"{args.min_identical_tokens} tokens ({ {b: divergence[b] for b in broken} }). "
              f"That is a wrong kernel, not rounding. Fix it or pass --allow-token-drift.")
        return 3
    drift = [label for label in labels[1:] if identity[label] != identity[labels[0]]]
    if drift:
        message = f"greedy tokens differ across arms: {drift} vs {labels[0]}"
        if drift_expected(args.setting):
            print(f"NOTE {message} (expected for {args.setting}: arms use different kernels; "
                  f"all arms match stock for the first {args.min_identical_tokens} tokens)")
        elif args.allow_token_drift:
            print(f"WARNING {message}")
        else:
            print(f"refusing to run: {message}. Fix the kernel or pass --allow-token-drift.")
            return 3
    else:
        print("token identity: all arms produce identical greedy output")

    clocks_before = device_clock_record()
    if clocks_before:
        print(f"clocks before: {clocks_before}")

    def report(label, offset, run):
        timing = run.step_timing
        print(f"  [{label} run {offset + 1}/{args.repeats}] "
              f"gap={timing.get('expected_gap_ms', 0):.2f}ms "
              f"decode={timing.get('decode_step_p50_ms', 0):.2f}ms "
              f"host={timing.get('host_stage_ms_p50', 0):.2f}ms "
              f"gpu={timing.get('decode_gpu_ms_p50', 0):.2f}ms "
              f"ttft={run.latency.get('ttft_p50', 0):.1f}ms")

    print(f"\n=== interleaved: {' / '.join(labels)} x {args.repeats} ===")
    arms = run_interleaved(makers, config, repeats=args.repeats, on_run=report)
    clocks_after = device_clock_record()

    for label, repeated in arms.items():
        print(f"\n=== {label} ===")
        for metric in ("step_timing.expected_gap_ms", "latency.itl_p50",
                       "latency.ttft_p50", "waste_ratio",
                       "mean_decode_batch", "mean_context_tokens",
                       "step_timing.decode_step_p50_ms", "step_timing.prefill_step_p50_ms",
                       "step_timing.prefill_step_fraction",
                       "step_timing.host_stage_ms_p50", "step_timing.decode_gpu_ms_p50",
                       "step_timing.prefill_gpu_ms_p50", "step_timing.sync_ms_p50"):
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

    # Every arm after the first is compared against the first. With two arms this is
    # the usual A/B; `loo_all` compares each leave-one-out arm against the shared `full`.
    # expected_gap_ms leads because it is the only latency metric sensitive to a change
    # in the *mix* of step kinds. A median over token gaps is not: when prefill rose from
    # 22.7% to 46.9% of steps, itl_p50 stayed inside noise because the median gap was
    # still a decode step, while the average gap worsened 49%.
    COMPARED = ("step_timing.expected_gap_ms", "step_timing.prefill_share_of_gap",
                "latency.itl_p50", "latency.itl_p99", "latency.itl_p999",
                "latency.ttft_p50", "step_timing.decode_step_p50_ms",
                "step_timing.prefill_step_p50_ms",
                "step_timing.prefill_step_fraction",
                "step_timing.prefill_penalty_p50_ms", "waste_ratio",
                "step_timing.host_stage_ms_p50", "step_timing.decode_gpu_ms_p50",
                "step_timing.prefill_gpu_ms_p50", "step_timing.sync_ms_p50")
    baseline = arms[labels[0]]
    comparisons = {}
    for label in labels[1:]:
        variant = arms[label]
        print(f"\n=== {label} vs {labels[0]} ===")
        comparisons[label] = {}
        for metric in COMPARED:
            verdict = _verdict(baseline.summary(metric), variant.summary(metric))
            comparisons[label][metric] = verdict
            print(f"  {metric:24s} {verdict}")
    comparison = comparisons[labels[1]]  # two-arm shape, kept for existing readers

    payload = {
        "setting": args.setting, "repeats": args.repeats,
        "config": {**config.__dict__, **shared},
        "arm_settings": arm_settings,
        "prompt_profile": args.prompt_profile,
        "mean_prompt_tokens": mean_prompt,
        "environment": environment_record(loaded.device),
        "git": git_record(),
        "clocks_before": clocks_before, "clocks_after": clocks_after,
        "token_identity": {"identical": not drift, "drifted_arms": drift,
                           "first_divergence_vs_stock": divergence},
        "order": "interleaved",
        "arms": {label: arm.to_dict() for label, arm in arms.items()},
        "comparison": comparison,
        "comparisons": comparisons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {out}")
    return 0 if all(arm.ok for arm in arms.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
