"""Prefill chunk/budget policy under open-loop Poisson arrivals, including overload.

The closed-loop sweep (`ab_prefill_budget_packed_chat_qwen3_1.7b.json`) found budgets
above the chunk dominated. With arrivals faster than service, prompts queue, and a larger
budget might drain the queue sooner (better TTFT) at an ITL cost. Arms are interleaved
on one model; workload, summaries and the page audit come from `int8_kv_pressure_ab`.

    python -m benchmarks.batching.open_loop_budget_ab --rate 8 --out results/rtx4060/open_loop_budget_chat_1.7b_r8.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

from benchmarks.batching.int8_kv_pressure_ab import PROFILES, make_schedule, page_audit, run_workload, summarize

ARMS = {"c128_b128": (128, 128), "c128_b256": (128, 256), "c64_b64": (64, 64)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--rate", type=float, default=8.0, help="Poisson arrivals per second")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model, dtype="float16")
    profile = dict(PROFILES["chat"], arrival_rate=args.rate, requests=args.requests, max_active=args.max_active)
    schedule = make_schedule(loaded.tokenizer, profile, 7)
    names = list(ARMS)
    runs = {name: [] for name in names}
    audits = {name: [] for name in names}
    for repeat in range(args.repeats):
        for name in names[repeat % len(names):] + names[:repeat % len(names)]:
            chunk, budget = ARMS[name]
            engine = ContinuousBatchingEngine(
                loaded.model, loaded.tokenizer, loaded.device, num_blocks=args.num_blocks,
                max_active=args.max_active, prefix_cache_blocks=0, packed_fused_prefill=True,
                prefill_chunk_size=chunk, max_prefill_tokens_per_iteration=budget,
                cuda_graph_batch_sizes=tuple(b for b in (1, 2, 4, 8) if b <= args.max_active),
            )
            engine.warmup()
            raw = run_workload(engine, schedule)
            audits[name].append(page_audit(engine, raw["requests"]))
            summary = summarize(raw, engine)
            runs[name].append(summary)
            print(f"[{name} r{repeat}] tok/s={summary['throughput_tok_s']:.0f} wall={summary['wall_s']:.1f}s "
                  f"itl p50/p99={summary['itl_ms']['p50']:.1f}/{summary['itl_ms']['p99']:.1f} "
                  f"ttft p50/p99={summary['ttft_ms']['p50']:.0f}/{summary['ttft_ms']['p99']:.0f} "
                  f"queue p95={summary['queue_total_ms']['p95']:.0f} preempt={summary['preemptions']}", flush=True)
            del engine, raw
            gc.collect()
            torch.cuda.empty_cache()

    def stat(name, path):
        values = []
        for run in runs[name]:
            value = run
            for part in path.split("."):
                value = value[part]
            values.append(value)
        median = statistics.median(values)
        return {"median": median, "spread": (max(values) - min(values)) / median if median else 0.0}

    metrics = ("throughput_tok_s", "wall_s", "itl_ms.p50", "itl_ms.p95", "itl_ms.p99", "ttft_ms.p50",
               "ttft_ms.p95", "ttft_ms.p99", "queue_total_ms.p95", "preemptions")
    medians = {name: {m: stat(name, m) for m in metrics} for name in names}
    for name in names[1:]:
        print(f"=== {name} vs {names[0]} ===")
        for m in metrics:
            base, var = medians[names[0]][m], medians[name][m]
            if not base["median"]:
                continue
            change = (var["median"] - base["median"]) / base["median"]
            noise = max(base["spread"], var["spread"])
            print(f"  {m:20s} {base['median']:9.4g} -> {var['median']:9.4g}  {change:+.1%} "
                  f"({'resolved' if abs(change) > noise else 'unresolved'}, spread {noise:.1%})")
    Path(args.out).write_text(json.dumps({
        "benchmark": "open_loop_budget_ab", "model": args.model, "profile": profile, "args": vars(args),
        "arms": {name: {"chunk": c, "budget": b} for name, (c, b) in ARMS.items()},
        "runs": runs, "medians": medians, "audits": audits, "gpu": torch.cuda.get_device_name(),
    }, indent=2, default=str))
    print(f"Saved -> {args.out}")
    return 1 if any(any(a) for a in audits.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
