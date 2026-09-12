"""M1 benchmark: cost of the paged read path vs stock sdpa attention.

This measures how much overhead the scatter/gather round trip adds to decode. That
overhead number is the *evidence* that later informs the M3 profiling step and the
decision about whether a Triton paged-attention kernel is justified.

Honest framing: M1's round trip is deliberately unoptimized (a Python loop scatter).
It is NOT meant to be fast — it is meant to prove correctness in the attention path.
The point of measuring it is to see the ceiling of naive gather cost, which motivates
either a vectorized gather (M2) or a fused kernel (M4).

Usage:
    python -m benchmarks.cache.paged_read_path \
        --prompt "Explain KV caching in one sentence." \
        --max-new-tokens 32 --block-sizes 8,16,32 \
        --warmup-runs 2 --runs 5 --output results/paged_read_path.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(props.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        })
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return info


def _time_generation(model, tokenizer, prompt, max_new_tokens, device):
    """Time a single greedy generation with CUDA events. Returns (tokens, decode_ms_list)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]

    torch.cuda.synchronize(device)
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    ev_start.record()
    with torch.inference_mode():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None,
        )
    ev_end.record()
    torch.cuda.synchronize(device)

    total_ms = ev_start.elapsed_time(ev_end)
    gen_tokens = out.shape[1] - prompt_len
    return gen_tokens, total_ms


def _run_config(model, tokenizer, prompt, max_new_tokens, device, warmup_runs, runs):
    for _ in range(warmup_runs):
        _time_generation(model, tokenizer, prompt, max_new_tokens, device)
    totals = []
    gen_tokens = 0
    for _ in range(runs):
        gen_tokens, total_ms = _time_generation(model, tokenizer, prompt, max_new_tokens, device)
        totals.append(total_ms)
    return {
        "generated_tokens": gen_tokens,
        "total_ms_mean": statistics.mean(totals),
        "total_ms_std": statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "tokens_per_sec": gen_tokens / (statistics.mean(totals) / 1000.0),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Explain KV caching in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-sizes", default="8,16,32")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="results/paged_read_path.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    device = torch.device("cuda")
    block_sizes = [int(b) for b in args.block_sizes.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.cache.paged_attention import (
        enable_paged_attention_on_model, get_paged_config,
    )

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    results = {"device_info": _device_info(), "config": vars(args), "runs": {}}

    # --- Baseline: stock sdpa ---
    print("\n=== Baseline (stock sdpa) ===")
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    baseline = _run_config(
        model, tokenizer, args.prompt, args.max_new_tokens, device,
        args.warmup_runs, args.runs,
    )
    results["runs"]["sdpa_baseline"] = baseline
    print(f"  {baseline['tokens_per_sec']:.1f} tok/s  "
          f"total={baseline['total_ms_mean']:.1f}ms (+/- {baseline['total_ms_std']:.1f})")

    # --- Paged read path, per block size ---
    for block_size in block_sizes:
        print(f"\n=== Paged read path (block_size={block_size}) ===")
        # verify=False for the timed runs (we already prove correctness in tests);
        # verification adds a .abs().max() per call that we don't want in the timing.
        enable_paged_attention_on_model(model, block_size_tokens=block_size, verify=False)
        res = _run_config(
            model, tokenizer, args.prompt, args.max_new_tokens, device,
            args.warmup_runs, args.runs,
        )
        cfg = get_paged_config()
        res["attention_calls"] = cfg.calls
        overhead = (res["total_ms_mean"] / baseline["total_ms_mean"] - 1.0) * 100
        res["overhead_pct_vs_baseline"] = overhead
        results["runs"][f"paged_block{block_size}"] = res
        print(f"  {res['tokens_per_sec']:.1f} tok/s  "
              f"total={res['total_ms_mean']:.1f}ms  "
              f"overhead=+{overhead:.1f}% vs sdpa")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("M1 Paged Read Path — Overhead vs Stock SDPA")
    print(f"{'='*60}")
    print(f"{'Config':<22} {'Tok/s':>8} {'Total ms':>10} {'Overhead':>10}")
    print("-" * 52)
    b = results["runs"]["sdpa_baseline"]
    print(f"{'sdpa_baseline':<22} {b['tokens_per_sec']:>8.1f} {b['total_ms_mean']:>10.1f} {'—':>10}")
    for block_size in block_sizes:
        r = results["runs"][f"paged_block{block_size}"]
        print(f"{'paged_block'+str(block_size):<22} {r['tokens_per_sec']:>8.1f} "
              f"{r['total_ms_mean']:>10.1f} {'+'+format(r['overhead_pct_vs_baseline'],'.1f')+'%':>10}")

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")
    print("\nNote: M1 round trip is intentionally unoptimized (Python-loop scatter).")
    print("This overhead is the naive-gather ceiling that motivates M2/M4 optimization.")


if __name__ == "__main__":
    main()
