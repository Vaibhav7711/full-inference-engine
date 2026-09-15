"""Paired end-to-end A/B for Qwen MLP gate/up projection fusion.

Both variants are warmed and CUDA-graph captured before measurement. Rounds alternate
their order so Colab clock/thermal drift cannot systematically favour one variant.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


PROMPTS = [
    "Explain how a transformer generates the next token.",
    "The capital city of Japan is",
    "Write a short explanation of paged attention.",
    "Two plus two equals",
]


def _run(engine, prompts: list[str], max_new_tokens: int) -> tuple[float, list[list[int]]]:
    engine.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    return time.perf_counter() - start, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cuda-graph-batch-size", type=int, default=16)
    parser.add_argument("--output", default="results/mlp_gate_up_fusion_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    if min(args.num_requests, args.max_new_tokens, args.rounds, args.num_blocks, args.block_size) <= 0:
        parser.error("all sizes must be positive")
    if args.cuda_graph_batch_size is not None and args.cuda_graph_batch_size != args.num_requests:
        parser.error("this paired benchmark requires graph width equal to num-requests")

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    prompts = [PROMPTS[index % len(PROMPTS)] for index in range(args.num_requests)]
    print("Loading separate unfused and fused model instances...")
    unfused_loaded, fused_loaded = load_model(args.model), load_model(args.model)
    common = dict(device="cuda", max_active=args.num_requests, num_blocks=args.num_blocks,
                  block_size=args.block_size, prefix_cache_blocks=0,
                  cuda_graph_batch_size=args.cuda_graph_batch_size)
    engines = {
        "unfused": ContinuousBatchingEngine(unfused_loaded.model, unfused_loaded.tokenizer,
                                             fuse_mlp_gate_up=False, **common),
        "fused": ContinuousBatchingEngine(fused_loaded.model, fused_loaded.tokenizer,
                                           fuse_mlp_gate_up=True, **common),
    }
    for engine in engines.values():
        engine.eos_ids.clear()

    # Compile and capture before timing. The output also establishes the greedy-token
    # equivalence baseline across independently loaded but identical model weights.
    print("Warming and capturing both variants...")
    expected = None
    for name, engine in engines.items():
        _, outputs = _run(engine, prompts, min(8, args.max_new_tokens))
        if expected is None:
            expected = outputs
        elif outputs != expected:
            raise RuntimeError(f"{name} changed greedy tokens during warmup")

    durations = {"unfused": [], "fused": []}
    print("\nPaired MLP gate/up fusion A/B")
    print(f"{'round':>6} {'first':>10} {'unfused ms':>13} {'fused ms':>11} {'fused speedup':>15}")
    reference = None
    for round_index in range(args.rounds):
        order = ("unfused", "fused") if round_index % 2 == 0 else ("fused", "unfused")
        round_values: dict[str, float] = {}
        for name in order:
            elapsed, outputs = _run(engines[name], prompts, args.max_new_tokens)
            if reference is None:
                reference = outputs
            elif outputs != reference:
                raise RuntimeError(f"{name} changed greedy tokens in round {round_index}")
            durations[name].append(elapsed * 1000.0)
            round_values[name] = elapsed * 1000.0
        speedup = round_values["unfused"] / round_values["fused"]
        print(f"{round_index + 1:>6} {order[0]:>10} {round_values['unfused']:>13.3f} "
              f"{round_values['fused']:>11.3f} {speedup:>14.3f}x")

    unfused_median, fused_median = statistics.median(durations["unfused"]), statistics.median(durations["fused"])
    result = {"config": vars(args), "unfused_ms": durations["unfused"], "fused_ms": durations["fused"],
              "unfused_median_ms": unfused_median, "fused_median_ms": fused_median,
              "fused_speedup": unfused_median / fused_median, "token_identical": True}
    print(f"\nMedian: unfused={unfused_median:.3f} ms  fused={fused_median:.3f} ms  "
          f"fused speedup={result['fused_speedup']:.3f}x; tokens identical")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
