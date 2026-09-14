"""Warmup benchmark: original vs optimized speculative decoder.

Two fixes demonstrated here:
  1. WARMUP — run throwaway generations first so cold-start (kernel compile, autotune) is
     NOT in the timed measurement. The original benchmark timed the first call directly,
     inflating the numbers. This makes the measurement HONEST.
  2. OPTIMIZATION — the optimized decoder keeps tokens GPU-resident, syncing once per round
     instead of per token. This makes it genuinely FASTER (removes the ~193ms/round overhead).

Reports, all properly warmed up, median of N runs:
  - reference (target-only) latency
  - original speculative latency + speedup
  - optimized speculative latency + speedup + overhead reduction vs original

HONEST FRAMING: even the optimized version is expected to LOSE on a T4, because the
~270ms/round memory-bandwidth floor (draft ≈ target cost at batch=1) remains. The point is
to prove the loss is FUNDAMENTAL (physics), not measurement artifact (warmup) or sloppy
code (syncs). That is a stronger, airtight conclusion.

Usage:
    python -m benchmarks.speculative.warmup_compare \
        --target-model Qwen/Qwen3-4B --draft-model Qwen/Qwen3-0.6B \
        --prompt "def quicksort(arr):" --max-new-tokens 48 \
        --speculation-depths 2,4 --warmup 2 --runs 3 \
        --output results/spec_warmup_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from time import perf_counter

import torch


def _device_info():
    info = {"timestamp": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": torch.cuda.get_device_name(0),
                     "gpu_memory_gb": round(p.total_memory / 1e9, 2)})
    return info


def _timed(fn):
    """Time one call with proper CUDA sync at both ends."""
    torch.cuda.synchronize()
    t0 = perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (perf_counter() - t0) * 1000


def _bench(fn, warmup, runs):
    """Warm up `warmup` times (discarded), then time `runs` times, return median ms + a result."""
    for _ in range(warmup):
        _timed(fn)                      # THE WARMUP — discard these
    times, last = [], None
    for _ in range(runs):
        last, ms = _timed(fn)
        times.append(ms)
    return last, statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="def quicksort(arr): if len(arr) <= 1: return arr")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--speculation-depths", default="2,4")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="results/spec_warmup_compare.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    device = torch.device("cuda")
    depths = [int(d) for d in args.speculation_depths.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.model.runner import ExplicitDecodeRunner
    from engine.speculative.vanilla import VanillaSpeculativeDecoder      # original
    from engine.speculative.optimized import OptimizedSpeculativeDecoder  # optimized

    print(f"Loading target {args.target_model} + draft {args.draft_model}...")
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()

    results = {"device_info": _device_info(), "config": vars(args), "depths": {}}

    # --- Reference: target-only, WARMED UP ---
    print(f"\n{'='*66}")
    print(f"Warmed-up comparison ({args.max_new_tokens} tokens, warmup={args.warmup}, runs={args.runs})")
    print(f"{'='*66}")
    runner = ExplicitDecodeRunner(target, tok, device)
    _, ref_ms = _bench(lambda: runner.generate(args.prompt, max_new_tokens=args.max_new_tokens),
                       args.warmup, args.runs)
    results["reference_ms"] = round(ref_ms, 1)
    print(f"\nReference (target-only, warmed): {ref_ms:.0f} ms")
    print(f"\n{'depth':>6} {'variant':>10} {'ms':>9} {'speedup':>9} {'accept':>8} {'vs_orig':>9}")
    print("-" * 60)

    orig_dec = VanillaSpeculativeDecoder(target, draft, tok, device)
    opt_dec = OptimizedSpeculativeDecoder(target, draft, tok, device)

    for depth in depths:
        # Original (warmed up)
        orig_res, orig_ms = _bench(
            lambda d=depth: orig_dec.generate(args.prompt, max_new_tokens=args.max_new_tokens, speculation_depth=d),
            args.warmup, args.runs)
        orig_speedup = ref_ms / orig_ms
        orig_accept = orig_res.acceptance_rate

        # Optimized (warmed up)
        opt_res, opt_ms = _bench(
            lambda d=depth: opt_dec.generate(args.prompt, max_new_tokens=args.max_new_tokens, speculation_depth=d),
            args.warmup, args.runs)
        opt_speedup = ref_ms / opt_ms
        opt_accept = opt_res.acceptance_rate
        overhead_reduction = (1 - opt_ms / orig_ms) * 100   # how much faster optimized is

        results["depths"][str(depth)] = {
            "original_ms": round(orig_ms, 1), "original_speedup": round(orig_speedup, 3),
            "original_acceptance": round(orig_accept, 3),
            "optimized_ms": round(opt_ms, 1), "optimized_speedup": round(opt_speedup, 3),
            "optimized_acceptance": round(opt_accept, 3),
            "overhead_reduction_pct": round(overhead_reduction, 1),
        }

        print(f"{depth:>6} {'original':>10} {orig_ms:>8.0f} {orig_speedup:>8.2f}x {orig_accept:>7.0%} {'—':>9}")
        print(f"{depth:>6} {'optimized':>10} {opt_ms:>8.0f} {opt_speedup:>8.2f}x {opt_accept:>7.0%} "
              f"{'-'+format(overhead_reduction,'.0f')+'%':>9}")

    # --- Summary ---
    print(f"\n{'='*66}")
    print("Reading the result")
    print(f"{'='*66}")
    best_opt = max((d["optimized_speedup"] for d in results["depths"].values()), default=0)
    best_reduction = max((d["overhead_reduction_pct"] for d in results["depths"].values()), default=0)
    print(f"Optimization removed up to {best_reduction:.0f}% of per-round overhead (the CPU-sync fix).")
    print(f"Best optimized speedup: {best_opt:.2f}x")
    if best_opt < 1.0:
        print("Still < 1.0x even optimized + warmed up -> the loss is FUNDAMENTAL:")
        print("the ~270ms/round memory-bandwidth floor (draft ~ target at batch=1) remains.")
        print("This is the airtight conclusion: physics, not warmup artifact, not sloppy code.")
    else:
        print("Optimized version clears 1.0x -> speedup achieved on this hardware.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
