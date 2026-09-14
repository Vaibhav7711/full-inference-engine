"""BS4: the speculation-batching crossover experiment.

THE QUESTION: does speculative decoding still help as batch size grows?

Batching gets its throughput by filling the GPU's idle compute with more sequences.
Speculation ALSO needs idle compute (to verify draft tokens cheaply). They compete for the
same resource. The literature predicts speculation's benefit shrinks as batch grows — this
benchmark MEASURES that on your own engine.

At each batch size N in {1, 2, 4, 8}:
    BASELINE:  batched greedy decode of the target (no speculation)
    SPECULATIVE: BatchedSpeculativeEngine at a fixed speculation depth
    Report throughput (tok/s) for each, and the ratio spec/greedy.
    ratio > 1.0 -> speculation helps at that batch size
    ratio < 1.0 -> speculation hurts (batching already saturates compute)
    The batch size where the ratio crosses 1.0 is THE crossover.

Everything is warmed up and uses median-of-N timing (the lessons from earlier).

Honest expectation on a T4: because a 0.6B draft is NOT cheap relative to the target at
batch=1 (memory-latency-bound), speculation may lose at EVERY batch size, and lose MORE as
batch grows. That is still a real, defensible measured result. We measure; we don't assume.

Usage:
    python -m benchmarks.batching.spec_batch_crossover \
        --target-model Qwen/Qwen3-1.7B --draft-model Qwen/Qwen3-0.6B \
        --batch-sizes 1,2,4,8 --max-new-tokens 32 --depth 4 \
        --warmup 1 --runs 3 --output results/spec_batch_crossover.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from time import perf_counter

import torch


PROMPT_POOL = [
    "The capital of France is",
    "Once upon a time in a land far away",
    "def fibonacci(n): if n <= 1: return n",
    "The theory of relativity states that",
    "Water is composed of hydrogen and",
    "To write a for loop in Python, you",
    "The largest planet in the solar system is",
    "Machine learning is a subfield of",
]


def _device_info():
    info = {"timestamp": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": torch.cuda.get_device_name(0),
                     "gpu_memory_gb": round(p.total_memory / 1e9, 2)})
    return info


def _timed(fn):
    torch.cuda.synchronize()
    t0 = perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, perf_counter() - t0


def _bench(fn, warmup, runs):
    for _ in range(warmup):
        _timed(fn)
    times, last = [], None
    for _ in range(runs):
        last, s = _timed(fn)
        times.append(s)
    return last, statistics.median(times)


def _batched_greedy(engine, prompts, max_new):
    """Plain batched greedy decode of the target — the no-speculation baseline.
    Uses the engine's own batched prefill/decode so the kernel path is identical."""
    tok, device = engine.tok, engine.device
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    cache, nxt, mask = engine._batched_prefill(engine.target, enc.input_ids, enc.attention_mask)
    N = len(prompts)
    outs = [[] for _ in range(N)]
    done = [False] * N
    for _ in range(max_new):
        toks = nxt.tolist()
        for i in range(N):
            if not done[i]:
                outs[i].append(toks[i][0])
                if toks[i][0] in engine.eos_ids:
                    done[i] = True
        if all(done):
            break
        nxt, cache, mask = engine._batched_decode(engine.target, nxt, cache, mask)
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="results/spec_batch_crossover.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    device = torch.device("cuda")
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.batching.batched_speculative import BatchedSpeculativeEngine

    print(f"Loading target {args.target_model} + draft {args.draft_model}...")
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()
    target.config._attn_implementation = "sdpa"
    draft.config._attn_implementation = "sdpa"

    engine = BatchedSpeculativeEngine(target, draft, tok, device)

    results = {"device_info": _device_info(), "config": vars(args), "sweep": []}

    print(f"\n{'='*72}")
    print(f"Speculation x Batching crossover  (depth={args.depth}, {args.max_new_tokens} tok/seq, "
          f"warmup={args.warmup}, runs={args.runs})")
    print(f"{'='*72}")
    print(f"{'batch':>6} {'greedy tok/s':>13} {'spec tok/s':>11} {'ratio':>7} {'accept':>8} {'verdict':>14}")
    print("-" * 66)

    for N in batch_sizes:
        prompts = [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(N)]

        # --- Baseline: batched greedy (no speculation) ---
        g_out, g_s = _bench(lambda: _batched_greedy(engine, prompts, args.max_new_tokens),
                            args.warmup, args.runs)
        g_tokens = sum(len(o) for o in g_out)
        g_tps = g_tokens / g_s

        # --- Speculative: batched spec ---
        s_res, s_s = _bench(lambda: engine.generate(prompts, max_new_tokens=args.max_new_tokens,
                                                    speculation_depth=args.depth),
                            args.warmup, args.runs)
        s_tokens = sum(len(o) for o in s_res.outputs)
        s_tps = s_tokens / s_s
        ratio = s_tps / g_tps if g_tps > 0 else 0.0
        verdict = "spec HELPS" if ratio > 1.0 else "spec hurts"

        results["sweep"].append({
            "batch_size": N,
            "greedy_tok_s": round(g_tps, 1), "greedy_s": round(g_s, 3),
            "spec_tok_s": round(s_tps, 1), "spec_s": round(s_s, 3),
            "spec_over_greedy": round(ratio, 3),
            "acceptance_rate": round(s_res.acceptance_rate, 3),
            "spec_rounds": s_res.total_rounds,
        })
        print(f"{N:>6} {g_tps:>13.1f} {s_tps:>11.1f} {ratio:>6.2f}x {s_res.acceptance_rate:>7.0%} {verdict:>14}")

    # --- Reading ---
    print(f"\n{'='*72}")
    print("Reading the crossover")
    print(f"{'='*72}")
    helps = [r for r in results["sweep"] if r["spec_over_greedy"] > 1.0]
    if not helps:
        print("Speculation never beats batched greedy at any tested batch size on this GPU.")
        print("Consistent with: draft not cheap relative to target (memory-latency-bound),")
        print("and batching already consuming the idle compute speculation would need.")
    else:
        best = max(helps, key=lambda r: r["spec_over_greedy"])
        cross = next((r for r in results["sweep"] if r["spec_over_greedy"] <= 1.0), None)
        print(f"Speculation helps up to batch {best['batch_size']} ({best['spec_over_greedy']:.2f}x).")
        if cross:
            print(f"Crossover: by batch {cross['batch_size']} it stops helping ({cross['spec_over_greedy']:.2f}x).")
        print("Batching consumes the idle compute speculation relies on; benefit shrinks with N.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
