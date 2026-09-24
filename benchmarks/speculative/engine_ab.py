"""Kaggle dual-T4 A/B: ordinary paged engine versus live draft speculation."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from time import perf_counter

import torch

from engine.model import load_model
from engine.speculative import DraftModelProposer


PROMPTS = [
    "Explain why the sky appears blue in three sentences.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "A train travels 120 km in 90 minutes. Explain its average speed step by step.",
    "Paged attention stores KV tensors in fixed-size blocks. Summarize the benefit.",
    "Repeat and explain: alpha beta gamma alpha beta gamma alpha beta gamma.",
    "What is the capital of France, and which river runs through it?",
    "Compare breadth-first search and depth-first search with one example.",
    "Complete this JSON pattern and explain it: {\"a\": 1, \"b\": 2,",
]


def _sync() -> None:
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def _run(engine, prompts: list[str], max_new_tokens: int) -> tuple[list[list[int]], float, dict]:
    engine.reset()
    before = engine.stats_snapshot()
    _sync()
    started = perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    _sync()
    elapsed_ms = (perf_counter() - started) * 1000
    after = engine.stats_snapshot()
    counters = {}
    for key in (
        "speculative_rounds", "speculative_proposed_tokens",
        "speculative_accepted_tokens", "speculative_fallbacks",
    ):
        counters[key] = int(after.get(key, 0)) - int(before.get(key, 0))
    proposed = counters["speculative_proposed_tokens"]
    counters["speculative_acceptance_rate"] = (
        counters["speculative_accepted_tokens"] / proposed if proposed else 0.0
    )
    return outputs, elapsed_ms, counters


def _measure(engine, prompts, max_new_tokens, warmup, runs) -> tuple[list[list[int]], list[float], dict]:
    for _ in range(warmup):
        _run(engine, prompts, min(max_new_tokens, 16))
    samples = [_run(engine, prompts, max_new_tokens) for _ in range(runs)]
    elapsed = [sample[1] for sample in samples]
    middle = statistics.median(elapsed)
    representative = min(samples, key=lambda sample: abs(sample[1] - middle))
    return representative[0], elapsed, representative[2]


def _engine(model, tokenizer, device, args, *, proposer=None):
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    buckets = tuple(int(value) for value in args.graph_buckets.split(",") if value)
    return ContinuousBatchingEngine(
        model, tokenizer, device, num_blocks=args.num_blocks, block_size=16,
        max_active=max(args.concurrencies), cuda_graph_batch_sizes=buckets or None,
        speculative_proposer=proposer, speculation_depth=args.depth,
        fused_step=proposer is None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--graph-buckets", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.concurrencies = tuple(int(value) for value in args.concurrencies.split(","))
    if torch.cuda.device_count() < 2:
        raise SystemExit("engine A/B requires two visible CUDA GPUs")
    if min(*args.concurrencies, args.depth, args.max_new_tokens, args.num_blocks, args.runs) <= 0:
        parser.error("concurrencies, depth, lengths, blocks and runs must be positive")

    target = load_model(args.target_model, dtype="float16", device="cuda:0")
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": args.target_model, "target_revision": target.resolved_revision,
        "draft_model": args.draft_model, "config": {
            key: (list(value) if isinstance(value, tuple) else str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "rows": [],
    }

    references: dict[int, list[list[int]]] = {}
    baseline_times: dict[int, list[float]] = {}
    baseline = _engine(target.model, target.tokenizer, target.device, args)
    baseline.warmup()
    for concurrency in args.concurrencies:
        prompts = [PROMPTS[index % len(PROMPTS)] for index in range(concurrency)]
        outputs, times, _ = _measure(
            baseline, prompts, args.max_new_tokens, args.warmup, args.runs,
        )
        references[concurrency], baseline_times[concurrency] = outputs, times
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    draft = load_model(args.draft_model, dtype="float16", device="cuda:1")
    if target.tokenizer.get_vocab() != draft.tokenizer.get_vocab():
        raise SystemExit("target and draft token maps differ")
    proposer = DraftModelProposer(draft.model, draft.device)
    speculative = _engine(
        target.model, target.tokenizer, target.device, args, proposer=proposer,
    )
    speculative.warmup()
    for concurrency in args.concurrencies:
        prompts = [PROMPTS[index % len(PROMPTS)] for index in range(concurrency)]
        outputs, spec_times, counters = _measure(
            speculative, prompts, args.max_new_tokens, args.warmup, args.runs,
        )
        baseline_ms = statistics.median(baseline_times[concurrency])
        speculative_ms = statistics.median(spec_times)
        result["rows"].append({
            "concurrency": concurrency,
            "baseline_ms": baseline_ms, "speculative_ms": speculative_ms,
            "speedup": baseline_ms / speculative_ms,
            "tokens_match": outputs == references[concurrency],
            "raw_baseline_ms": baseline_times[concurrency],
            "raw_speculative_ms": spec_times,
            **counters,
        })

    result["engine_stats"] = speculative.stats_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(row["tokens_match"] for row in result["rows"]):
        raise SystemExit("live engine A/B found a token mismatch")
    if not any(row["speedup"] > 1 for row in result["rows"]):
        raise SystemExit("correct pair produced no measured live-engine speedup")


if __name__ == "__main__":
    main()
