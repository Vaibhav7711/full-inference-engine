"""Sweep decode-first prefill budgets on the reproducible mixed-arrival workload."""

from __future__ import annotations

import argparse
import json
import os

import torch

from benchmarks.batching.mixed_arrival_workload import _make_schedule, _run, _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--budgets", default="64,128,256")
    parser.add_argument("--long-repeats", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-active", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--graph-buckets", default="2,4,8,16")
    parser.add_argument("--output", default="results/mixed_arrival_phase11_prefill_budget_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    budgets = [int(value) for value in args.budgets.split(",") if value]
    buckets = tuple(int(value) for value in args.graph_buckets.split(",") if value)
    if not budgets or min(*budgets, args.prefill_chunk_size) <= 0 or max(buckets, default=0) > args.max_active:
        parser.error("budgets/chunk size must be positive and graph buckets must fit max-active")

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, "cuda", max_active=args.max_active,
        num_blocks=args.num_blocks, prefill_chunk_size=args.prefill_chunk_size,
        max_prefill_tokens_per_iteration=max(budgets), cuda_graph_batch_sizes=buckets,
    )
    schedule = _make_schedule(loaded.tokenizer, args.long_repeats, args.max_new_tokens)
    rows = []
    print("\nMixed-arrival prefill-budget A/B")
    print(f"{'budget':>8} {'tok/s':>9} {'short p95 ITL':>14} {'medium p95 ITL':>15} {'long p50 TTFT':>14} {'long p95 TTFT':>14}")
    for budget in budgets:
        engine.max_prefill_tokens_per_iteration = budget
        # First run compiles/captures any path absent from previous variants.
        _run(engine, schedule, min(4, args.max_new_tokens))
        elapsed, requests = _run(engine, schedule, args.max_new_tokens)
        by_class = _summary(requests)
        tokens = sum(len(request.output_token_ids) for _, request in requests)
        row = {"prefill_token_budget": budget, "elapsed_s": elapsed,
               "throughput_tok_s": tokens / elapsed, "by_class": by_class}
        rows.append(row)
        print(f"{budget:>8} {row['throughput_tok_s']:>9.1f} {by_class['short']['itl_p95_ms']:>14.2f} "
              f"{by_class['medium']['itl_p95_ms']:>15.2f} {by_class['long']['ttft_p50_ms']:>14.2f} "
              f"{by_class['long']['ttft_p95_ms']:>14.2f}")
    result = {"config": vars(args), "rows": rows}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
