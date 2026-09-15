"""Reproducible mixed-arrival serving workload for scheduler decisions.

The workload introduces short and long prompts over scheduler iterations instead of
submitting everything at time zero. It reports aggregate throughput plus request-level
queue time, TTFT, and decode inter-token latency (ITL), grouped by request class.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from engine.runtime import GenerationRequest


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(requests: list[tuple[str, GenerationRequest]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[GenerationRequest]] = {}
    for group, request in requests:
        groups.setdefault(group, []).append(request)
    result = {}
    for group, rows in groups.items():
        queue = [item.queue_time_ms() or 0.0 for item in rows]
        ttft = [item.time_to_first_token_ms() or 0.0 for item in rows]
        itl = [
            (right - left) / 1_000_000
            for item in rows
            for left, right in zip(item.token_timestamps_ns, item.token_timestamps_ns[1:])
        ]
        result[group] = {
            "requests": len(rows), "queue_p50_ms": statistics.median(queue),
            "queue_p95_ms": _percentile(queue, 0.95), "ttft_p50_ms": statistics.median(ttft),
            "ttft_p95_ms": _percentile(ttft, 0.95), "itl_p50_ms": statistics.median(itl) if itl else 0.0,
            "itl_p95_ms": _percentile(itl, 0.95), "itl_max_ms": max(itl, default=0.0),
        }
    return result


def _make_schedule(tokenizer, long_repeats: int, max_new_tokens: int) -> list[tuple[int, str, list[int]]]:
    short = [
        "The capital of France is", "Two plus two equals", "Explain KV caching briefly.",
        "The largest planet is", "A Python list is", "The speed of light is",
    ]
    medium = [
        "Explain why continuous batching improves inference throughput in a concise paragraph. ",
        "Describe a robust scheduler for an LLM inference server with bounded memory. ",
    ]
    long = (
        "Describe a production GPU inference runtime with paging, scheduling, cancellation, "
        "continuous batching, prefix caching, and observability. "
        * long_repeats
    )
    rows: list[tuple[int, str, str]] = []
    rows += [(0, "short", text) for text in short[:4]]
    rows += [(0, "medium", text * 4) for text in medium]
    rows += [(3, "long", long), (3, "long", long)]
    rows += [(7, "short", text) for text in short[4:]]
    rows += [(11, "medium", text * 4) for text in medium]
    return [
        (arrival, group, tokenizer(text, return_tensors="pt").input_ids[0].tolist())
        for arrival, group, text in rows
    ]


def _run(engine, schedule, max_new_tokens: int) -> tuple[float, list[tuple[str, GenerationRequest]]]:
    engine.reset()
    engine.eos_ids.clear()  # fixed work makes the latency distribution comparable.
    submitted: list[tuple[str, GenerationRequest]] = []
    cursor, tick = 0, 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    while cursor < len(schedule) or engine.has_unfinished_requests:
        while cursor < len(schedule) and schedule[cursor][0] <= tick:
            _, group, token_ids = schedule[cursor]
            request = GenerationRequest(
                request_id=f"mixed-{cursor}", prompt_token_count=len(token_ids),
                max_new_tokens=max_new_tokens, prompt_token_ids=token_ids,
            )
            if not engine.submit(request):
                raise RuntimeError(f"scheduler rejected {request.request_id}")
            submitted.append((group, request))
            cursor += 1
        engine.step()
        tick += 1
    torch.cuda.synchronize()
    return time.perf_counter() - started, submitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--long-repeats", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-active", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--prefill-token-budget", type=int, default=256)
    parser.add_argument("--graph-buckets", default="2,4,8,16")
    parser.add_argument("--output", default="results/mixed_arrival_phase11_baseline.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    buckets = tuple(int(value) for value in args.graph_buckets.split(",") if value)
    if not buckets or max(buckets) > args.max_active:
        parser.error("graph buckets must be non-empty and no larger than max-active")

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, "cuda", max_active=args.max_active,
        num_blocks=args.num_blocks, prefill_chunk_size=args.prefill_chunk_size,
        max_prefill_tokens_per_iteration=args.prefill_token_budget,
        cuda_graph_batch_sizes=buckets,
    )
    schedule = _make_schedule(loaded.tokenizer, args.long_repeats, args.max_new_tokens)
    # Compile each relevant path and graph before the measured run.
    _run(engine, schedule, min(4, args.max_new_tokens))
    elapsed, requests = _run(engine, schedule, args.max_new_tokens)
    total_tokens = sum(len(request.output_token_ids) for _, request in requests)
    result = {"config": vars(args), "elapsed_s": elapsed, "total_output_tokens": total_tokens,
              "throughput_tok_s": total_tokens / elapsed, "by_class": _summary(requests),
              "token_identical_not_applicable": True}
    print("\nMixed-arrival scheduler baseline")
    print(f"total output: {total_tokens} tokens in {elapsed:.3f}s ({result['throughput_tok_s']:.1f} tok/s)")
    print(f"{'class':>8} {'count':>6} {'queue p50':>11} {'TTFT p50':>11} {'TTFT p95':>11} {'ITL p50':>11} {'ITL p95':>11} {'ITL max':>11}")
    for group, row in result["by_class"].items():
        print(f"{group:>8} {row['requests']:>6} {row['queue_p50_ms']:>10.2f} {row['ttft_p50_ms']:>10.2f} "
              f"{row['ttft_p95_ms']:>10.2f} {row['itl_p50_ms']:>10.2f} {row['itl_p95_ms']:>10.2f} {row['itl_max_ms']:>10.2f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
