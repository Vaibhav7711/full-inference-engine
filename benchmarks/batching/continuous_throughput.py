"""D4: continuous-batching throughput — the payoff measurement.

Measures the end-to-end throughput win of continuous batching. Unlike paging (a memory
win, invisible at batch=1) and quantization (a memory win, actually slower), continuous
batching is a GENUINE throughput win: N sequences share each forward pass, so the model
weights are read from memory once per step instead of once per sequence.

Apples-to-apples comparison, both using the SAME engine and kernel:
    SEQUENTIAL: run each request one at a time (the engine with max_active=1, so no
                batching — same prefill, same K4 decode, just one sequence per forward).
    BATCHED:    run all requests together (max_active=concurrency), sharing forward passes.

    speedup = sequential_total_time / batched_total_time, swept over concurrency.

This isolates the batching win: same code path, same kernel, the only difference is
whether sequences share forward passes. That's the honest measurement.

What this proves: continuous batching gives an X times throughput improvement at
concurrency N, because batched forward passes amortize the per-step weight reads across
all active sequences.

Interview one-liner: "I measured my continuous-batching engine against sequential
processing — batching N sequences gave an X times end-to-end throughput improvement,
because the model weights are read once per step and amortized across all sequences
instead of re-read for each one."

Usage:
    python -m benchmarks.batching.continuous_throughput \
        --num-requests 32 --max-new-tokens 32 --concurrencies 1,2,4,8,16 \
        --output results/continuous_throughput.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch


def _device_info() -> dict:
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(p.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        })
    return info


# A pool of varied prompts (different lengths -> realistic mixed workload)
PROMPT_POOL = [
    "The capital of France is",
    "Once upon a time in a distant land",
    "2 + 2 =",
    "The transformer architecture works by",
    "Explain photosynthesis in simple terms.",
    "The largest planet in the solar system is",
    "To write a for loop in Python, you",
    "The theory of relativity states that",
    "Water is composed of hydrogen and",
    "The capital city of Japan is",
    "Machine learning is a subfield of",
    "The speed of light in a vacuum is approximately",
    "A binary search algorithm works by repeatedly",
    "The mitochondria is known as the",
    "In economics, supply and demand describe",
    "The French Revolution began in the year",
]


def _make_requests(n: int) -> list[str]:
    return [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(n)]


def _run_timed(engine, prompts, max_new_tokens, max_active):
    """Run prompts through the engine with a given max_active, return (elapsed_s, total_out_tokens)."""
    engine.reset()   # clean allocator slate for a fair, repeatable run
    engine.max_active = max_active
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(o) for o in outputs)
    return elapsed, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cuda-graph-batch-size", type=int, default=None,
                        help="enable paged CUDA-Graph replay only at this fixed active width")
    parser.add_argument("--enable-mlp-gate-up-fusion", action="store_true",
                        help="enable the experimental fused Qwen MLP gate/up projection")
    parser.add_argument("--warmup", action="store_true", help="run a small warmup first")
    parser.add_argument("--output", default="results/continuous_throughput.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    device = "cuda"
    concurrencies = [int(c) for c in args.concurrencies.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    prompts = _make_requests(args.num_requests)

    results = {"device_info": _device_info(), "config": vars(args), "sweep": []}

    # ONE engine, reused across all concurrency levels (the pool is large; don't make many).
    engine = ContinuousBatchingEngine(model, tok, device,
                                      num_blocks=args.num_blocks, block_size=args.block_size,
                                      cuda_graph_batch_size=args.cuda_graph_batch_size,
                                      fuse_mlp_gate_up=args.enable_mlp_gate_up_fusion)

    # Warmup (compile kernels, warm caches)
    if args.warmup:
        print("Warmup...")
        _run_timed(engine, prompts[:4], 8, max_active=4)

    # --- SEQUENTIAL baseline: max_active=1 (one sequence at a time, same engine) ---
    print(f"\n{'='*66}")
    print(f"Continuous Batching Throughput  ({args.num_requests} requests, "
          f"{args.max_new_tokens} tokens each)")
    print(f"{'='*66}")

    seq_time, seq_tokens = _run_timed(engine, prompts, args.max_new_tokens, max_active=1)
    seq_throughput = seq_tokens / seq_time
    results["sequential"] = {
        "max_active": 1,
        "elapsed_s": round(seq_time, 3),
        "total_tokens": seq_tokens,
        "throughput_tok_s": round(seq_throughput, 1),
    }

    print(f"\n{'concurrency':>12} {'elapsed_s':>10} {'tok/s':>10} {'speedup':>10}")
    print("-" * 46)
    print(f"{'1 (seq)':>12} {seq_time:>10.3f} {seq_throughput:>10.1f} {'1.00x':>10}")

    # --- BATCHED: increasing concurrency ---
    for c in concurrencies:
        if c == 1:
            continue  # already have sequential
        elapsed, tokens = _run_timed(engine, prompts, args.max_new_tokens, max_active=c)
        throughput = tokens / elapsed
        speedup = seq_time / elapsed
        results["sweep"].append({
            "concurrency": c,
            "elapsed_s": round(elapsed, 3),
            "total_tokens": tokens,
            "throughput_tok_s": round(throughput, 1),
            "speedup_vs_sequential": round(speedup, 3),
        })
        print(f"{c:>12} {elapsed:>10.3f} {throughput:>10.1f} {format(speedup,'.2f')+'x':>10}")

    # --- Save + summary ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")

    if results["sweep"]:
        best = max(results["sweep"], key=lambda r: r["speedup_vs_sequential"])
        print(f"\nBest: {best['speedup_vs_sequential']:.2f}x throughput at concurrency "
              f"{best['concurrency']} "
              f"({best['throughput_tok_s']:.0f} tok/s vs {seq_throughput:.0f} sequential).")
    print("\nHonest claim: continuous batching gives an end-to-end throughput improvement")
    print("that grows with concurrency, because batched forward passes amortize the")
    print("per-step model-weight reads across all active sequences.")


if __name__ == "__main__":
    main()
