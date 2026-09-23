"""End-to-end occupancy sweep for padded CUDA-Graph decode buckets."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def _timed(engine, prompts: list[str], max_new_tokens: int) -> tuple[float, list[list[int]]]:
    engine.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    return time.perf_counter() - start, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--occupancies", default="1,2,3,4,8,16")
    parser.add_argument("--graph-buckets", default="2,4,8,16,32")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--output", default="results/padded_graph_occupancy_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    occupancies = [int(value) for value in args.occupancies.split(",") if value]
    buckets = tuple(int(value) for value in args.graph_buckets.split(",") if value)
    if not occupancies or not buckets or min(*occupancies, *buckets, args.max_new_tokens) <= 0:
        parser.error("occupancies, buckets, and max-new-tokens must be positive")
    max_active = max(max(occupancies), max(buckets))

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    normal = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, "cuda", max_active=max_active,
                                      num_blocks=args.num_blocks, prefix_cache_blocks=0)
    graphed = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, "cuda", max_active=max_active,
                                       num_blocks=args.num_blocks, prefix_cache_blocks=0,
                                       cuda_graph_batch_sizes=buckets)
    # Prevent an EOS from changing the amount of decode work between paths.
    normal.eos_ids.clear(); graphed.eos_ids.clear()
    base_prompts = [
        "Paged graph bucket request number zero.", "Explain paged KV caching briefly.",
        "The transformer uses attention to", "2 plus 2 equals",
    ]
    rows = []
    print("\nPadded CUDA-Graph occupancy A/B")
    print(f"{'live':>6} {'normal tok/s':>14} {'graph tok/s':>13} {'speedup':>9} {'tokens':>8}")
    for live in occupancies:
        prompts = [base_prompts[index % len(base_prompts)] for index in range(live)]
        # Warm each exact graph bucket; capture cost is deliberately excluded.
        _timed(graphed, prompts, min(4, args.max_new_tokens))
        normal_s, normal_out = _timed(normal, prompts, args.max_new_tokens)
        graph_s, graph_out = _timed(graphed, prompts, args.max_new_tokens)
        tokens = sum(len(row) for row in normal_out)
        normal_tps, graph_tps = tokens / normal_s, tokens / graph_s
        row = {"live_requests": live, "normal_seconds": normal_s, "graph_seconds": graph_s,
               "normal_tokens_per_second": normal_tps, "graph_tokens_per_second": graph_tps,
               "speedup": normal_s / graph_s, "token_identical": normal_out == graph_out}
        rows.append(row)
        print(f"{live:>6} {normal_tps:>14.1f} {graph_tps:>13.1f} {row['speedup']:>8.2f}x {tokens:>8}")
        if not row["token_identical"]:
            raise RuntimeError(f"graph output diverged at occupancy {live}")
    result = {"config": vars(args), "rows": rows}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
