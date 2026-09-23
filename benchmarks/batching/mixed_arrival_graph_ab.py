"""End-to-end CUDA-graph A/B on the mixed-arrival serving workload."""

from __future__ import annotations

import argparse
import json
import os

import torch

from benchmarks.batching.mixed_arrival_workload import _make_schedule, _run, _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--long-repeats", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-active", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--prefill-token-budget", type=int, default=128)
    parser.add_argument("--graph-buckets", default="2,4,8,16")
    parser.add_argument("--output", default="results/mixed_arrival_phase12_graph_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    buckets = tuple(int(value) for value in args.graph_buckets.split(",") if value)
    if not buckets or max(buckets) > args.max_active:
        parser.error("graph buckets must be non-empty and no larger than max-active")

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    common = dict(max_active=args.max_active, num_blocks=args.num_blocks,
                  prefill_chunk_size=args.prefill_chunk_size,
                  max_prefill_tokens_per_iteration=args.prefill_token_budget,
                  prefix_cache_blocks=0)
    ordinary = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, "cuda", **common)
    graphed = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, "cuda",
                                       cuda_graph_batch_sizes=buckets, **common)
    schedule = _make_schedule(loaded.tokenizer, args.long_repeats, args.max_new_tokens)
    # Warm all ordinary, paged-prefill, and graph-capture paths outside timing.
    _run(ordinary, schedule, min(4, args.max_new_tokens))
    _run(graphed, schedule, min(4, args.max_new_tokens))
    ordinary_s, ordinary_requests = _run(ordinary, schedule, args.max_new_tokens)
    graph_s, graph_requests = _run(graphed, schedule, args.max_new_tokens)
    ordinary_tokens = [request.output_token_ids for _, request in ordinary_requests]
    graph_tokens = [request.output_token_ids for _, request in graph_requests]
    if ordinary_tokens != graph_tokens:
        raise RuntimeError("CUDA graph workload changed greedy output tokens")
    total_tokens = sum(len(tokens) for tokens in ordinary_tokens)
    result = {
        "config": vars(args), "total_output_tokens": total_tokens,
        "ordinary": {"elapsed_s": ordinary_s, "throughput_tok_s": total_tokens / ordinary_s,
                     "by_class": _summary(ordinary_requests)},
        "graphed": {"elapsed_s": graph_s, "throughput_tok_s": total_tokens / graph_s,
                    "by_class": _summary(graph_requests)},
        "speedup": ordinary_s / graph_s, "token_identical": True,
    }
    print("\nMixed-arrival CUDA-graph A/B")
    print(f"ordinary: {result['ordinary']['throughput_tok_s']:.1f} tok/s in {ordinary_s:.3f}s")
    print(f"graphed:  {result['graphed']['throughput_tok_s']:.1f} tok/s in {graph_s:.3f}s")
    print(f"speedup:  {result['speedup']:.2f}x; tokens identical")
    print(f"{'class':>8} {'ordinary TTFT p95':>19} {'graph TTFT p95':>17} {'ordinary ITL p95':>18} {'graph ITL p95':>16}")
    for group in result["ordinary"]["by_class"]:
        base, graph = result["ordinary"]["by_class"][group], result["graphed"]["by_class"][group]
        print(f"{group:>8} {base['ttft_p95_ms']:>19.2f} {graph['ttft_p95_ms']:>17.2f} "
              f"{base['itl_p95_ms']:>18.2f} {graph['itl_p95_ms']:>16.2f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
